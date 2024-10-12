from MAPPOAgentPopulations import MAPPOAgentPopulations
import torch
from torch import optim, nn
import torch.distributions as dist
from Network import fc_network
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math

class NAgentLBRDivAgentPopulations(MAPPOAgentPopulations):
    """
        A class that implements a MAPPO-based TrajeDi implementation.
        Adds additional constrained objectives on top of MAPPO. 
    """
     
    def __init__(self, obs_size, num_agents, num_populations, configs, act_sizes, device, logger, mode="train"):
        super().__init__(obs_size, num_agents, num_populations, configs, act_sizes, device, logger, mode)
        critic_dims = [self.num_agents * (self.obs_size + self.num_agents), *self.config.model.critic_dims, 1]

        # Add the slightly different critic network. 
        # Two critics for all populations (each shared by entire population)
        init_ortho = self.config.model.get("init_ortho", True)
        self.joint_action_value_functions = [fc_network(critic_dims, init_ortho).double().to(self.device) for _ in range(self.num_agents)]
        self.target_joint_action_value_functions = [fc_network(critic_dims, init_ortho).double().to(self.device) for _ in range(self.num_agents)]
        self.hard_copy()

        # Add lagrange multiplier to help uphold constraints
        lagrange_mat1 = self.config.train["init_lagrange"]*torch.ones([self.config.populations["num_populations"], self.config.populations["num_populations"]]).double().to(self.device)
        lagrange_mat1.fill_diagonal_(0.5)
        self.lagrange_multiplier_matrix1 = Variable(lagrange_mat1.data, requires_grad=False)

        # Add weight normalizing terms to ensure constant balance between Lagrange loss and entropy maximization
        self.normalizer1 = self.compute_const_lagrange1().mean(dim=-1, keepdim=False)
        self.normalizer2 = self.compute_const_lagrange1().mean(dim=0, keepdim=False)

        critic_params_list = [
            *(param for critic in self.joint_action_value_functions for param in critic.parameters())
        ]
        self.critic_optimizer = optim.Adam(
            critic_params_list,
            lr=self.config.train["lr"]
        )

        self.constant_lagrange = False
        self.conf_opt = True
        self.dual_constraints = False
        if self.config.env["name"] == "BRDiv":
            self.constant_lagrange = True
        elif self.config.env["name"] == "LBRDiv-Ego-Opt":
            self.conf_opt = False
        elif self.config.env["name"] == "LBRDiv-Two-Constraints":
            self.dual_constraints = True
            # Add lagrange multiplier to help uphold second constraints
            lagrange_mat2 = self.config.train["init_lagrange"]*torch.ones([self.config.populations["num_populations"], self.config.populations["num_populations"]]).double().to(self.device)
            lagrange_mat2.fill_diagonal_(0.5)
            self.lagrange_multiplier_matrix2 = Variable(lagrange_mat2.data, requires_grad=False)
            self.normalizer2 = self.compute_const_lagrange2().mean(dim=-1, keepdim=False)

    def compute_const_lagrange1(self):
        """
        A method that returns Lagrange multipliers related to first constraints
        :return: \alpha^{1}
        """
        return F.relu(self.lagrange_multiplier_matrix1)
    
    def compute_const_lagrange2(self):
        """
        A method that returns Lagrange multipliers related to first constraints
        :return: \alpha^{1}
        """
        return F.relu(self.lagrange_multiplier_matrix2)
    
    def compute_sp_advantages(self, obs, n_obs, acts, dones, trunc, rews):
        """
        A function that computes the weighted advantage function as described in Expression 14
        :param obs: Agent observation
        :param n_obs: Agent next observation
        :param acts: Agent action
        :param dones: Agent done flag
        :param rews: Agent reward flag
        :return opt_diversity_values.detach() - baseline_diversity_values.detach(): weighted advantage function
        :return all_entropy_weight1.detach(): Variable entropy weights given \alpha^1
        :return all_entropy_weight1.detach(): Variable entropy weights given \alpha^2
        """
        batch_size = obs.size()[0]
        obs_length = obs.size()[1]

        # Initialize empty lists that saves every important data for actor loss computation
        baseline_xp_matrics1 = []
        opt_xp_matrics1 = []
        baseline_xp_matrics2 = []
        opt_xp_matrics2 = []
        lagrangian_matrices11 = []
        lagrangian_matrices12 = []
        entropy_weight1 = []
        entropy_weight2 = []

        lagrange_matrix_mean_norm1 = self.compute_const_lagrange1().mean(dim=-1, keepdim=False)
        if self.dual_constraints:
            lagrange_matrix_mean_norm2 = self.compute_const_lagrange2().mean(dim=-1, keepdim=False)
        else:
            lagrange_matrix_mean_norm2 = self.compute_const_lagrange1().mean(dim=0, keepdim=False)

        pos_entropy_weights1 = (lagrange_matrix_mean_norm1/self.normalizer1) * self.config.loss_weights["entropy_regularizer_loss"]
        pos_entropy_weights2 = (lagrange_matrix_mean_norm2/self.normalizer2) * self.config.loss_weights["entropy_regularizer_loss"]

        # Initialize target diversity at end of episode to None
        for idx in reversed(range(obs_length)):
            obs_idx = torch.cat(
                [obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)
            n_obs_idx = torch.cat(
                [n_obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)],
                dim=-1)

            
            sp_rl_rews = rews[:, idx, :]
            sp_rl_done = dones[:, idx]
            sp_rl_trunc = trunc[:, idx]

            futures = [
                torch.jit.fork(model, n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1)) for model
                in self.joint_action_value_functions
            ]
            results = [torch.jit.wait(fut) for fut in futures]

            all_conf_target_diversity_values_trunced = results[:-1]
            all_conf_target_diversity_values_trunced = torch.cat(all_conf_target_diversity_values_trunced, dim=-1)
            target_diversity_value2_trunced = results[-1]

            if idx == obs_length - 1:
                all_conf_target_diversity_values = all_conf_target_diversity_values_trunced
                target_diversity_value2 = target_diversity_value2_trunced

            if not self.dual_constraints:
                lagrangian_matrix11 = self.compute_const_lagrange1().detach().clone()
                lagrangian_matrix11.fill_diagonal_(0.0)
                lagrangian_matrix11 = lagrangian_matrix11.unsqueeze(0).repeat(batch_size, 1, 1)
                
                lagrangian_matrix12 = self.compute_const_lagrange1().detach().clone()
                lagrangian_matrix12.fill_diagonal_(1.0) 
                lagrangian_matrix12 = lagrangian_matrix12.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                lagrangian_matrix11 = self.compute_const_lagrange1().detach().clone()
                lagrangian_matrix11.fill_diagonal_(0.5)
                lagrangian_matrix11 = lagrangian_matrix11.unsqueeze(0).repeat(batch_size, 1, 1)
                
                lagrangian_matrix12 = self.compute_const_lagrange2().detach().clone()
                lagrangian_matrix12.fill_diagonal_(0.5) 
                lagrangian_matrix12 = lagrangian_matrix12.unsqueeze(0).repeat(batch_size, 1, 1)


            offset = self.num_populations + self.num_agents
            accessed_index = obs_idx[:, :, -offset:-offset + self.num_populations].argmax(dim=-1)

            idx_confs, idx_egos = accessed_index[:, :-1], accessed_index[:, -1]

            all_conf_target_diversity_values[sp_rl_trunc==1] = all_conf_target_diversity_values_trunced[sp_rl_trunc==1]
            all_conf_target_diversity_values = (
                    sp_rl_rews[:,:-1] + (
                    self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1).repeat(1, self.num_agents-1)) * all_conf_target_diversity_values)
            ).detach()

            target_diversity_value2[sp_rl_trunc==1] = target_diversity_value2_trunced[sp_rl_trunc==1]
            target_diversity_value2 = (
                    sp_rl_rews[:,-1].view(-1, 1) + (
                    self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_diversity_value2)
            ).detach()

            futures = [
                torch.jit.fork(model, obs_idx.view(obs_idx.size(0), 1, -1).squeeze(1)) for model
                in self.joint_action_value_functions
            ]
            results = [torch.jit.wait(fut) for fut in futures]

            baseline_diversity_values1 = results[:-1]
            baseline_diversity_values1 = torch.cat(baseline_diversity_values1, dim=-1)
            baseline_diversity_values2 = results[-1]

            opt_xp_matrics1.append(all_conf_target_diversity_values.clone())
            baseline_xp_matrics1.append(baseline_diversity_values1)
            opt_xp_matrics2.append(target_diversity_value2.clone())
            baseline_xp_matrics2.append(baseline_diversity_values2)

            entropy_weight1.append(pos_entropy_weights1[idx_confs[:,0]])
            entropy_weight2.append(pos_entropy_weights2[idx_egos])

            lagrangian_matrices11.append(lagrangian_matrix11[torch.arange(obs_idx.size(0)), idx_confs[:,0], :])
            if self.dual_constraints:
                lagrangian_matrices12.append(lagrangian_matrix12[torch.arange(obs_idx.size(0)), idx_egos, :])
            else:
                lagrangian_matrices12.append(lagrangian_matrix12[torch.arange(obs_idx.size(0)), idx_confs[:,0], :])
                
        # Combine lists into a single tensor for actor loss from SP data
        all_baseline_matrices1 = torch.cat(baseline_xp_matrics1, dim=0)
        all_opt_matrices1 = torch.cat(opt_xp_matrics1, dim=0)
        all_baseline_matrices2 = torch.cat(baseline_xp_matrics2, dim=0)
        all_opt_matrices2 = torch.cat(opt_xp_matrics2, dim=0)
        all_lagrangian_matrices11 = torch.cat(lagrangian_matrices11, dim=0)
        all_lagrangian_matrices12 = torch.cat(lagrangian_matrices12, dim=0)
        all_entropy_weight1 = torch.cat(entropy_weight1, dim=0).unsqueeze(dim=-1).repeat(1, self.num_agents-1)
        all_entropy_weight2 = torch.cat(entropy_weight2, dim=0)

        weights1 = torch.sum(all_lagrangian_matrices11, dim=-1)
        weights2 = torch.sum(all_lagrangian_matrices12, dim=-1)

        if not self.dual_constraints:
            if not self.conf_opt:
                baseline_diversity_values1 = torch.ones_like(weights1).unsqueeze(-1).repeat(1, self.num_agents-1)*all_baseline_matrices2.repeat(1, self.num_agents-1) + weights1.unsqueeze(-1).repeat(1, self.num_agents-1) * all_baseline_matrices2.repeat(1, self.num_agents-1)
                opt_diversity_values1 = torch.ones_like(weights1).unsqueeze(-1).repeat(1, self.num_agents-1)*all_opt_matrices2.repeat(1, self.num_agents-1) + weights1.unsqueeze(-1).repeat(1, self.num_agents-1) * all_opt_matrices2.repeat(1, self.num_agents-1)
            else:                                       
                baseline_diversity_values1 = torch.ones_like(weights1).unsqueeze(-1).repeat(1, self.num_agents-1)*all_baseline_matrices1 + weights1.unsqueeze(-1).repeat(1, self.num_agents-1) * all_baseline_matrices2.repeat(1, self.num_agents-1)
                opt_diversity_values1 = torch.ones_like(weights1).unsqueeze(-1).repeat(1, self.num_agents-1)*all_opt_matrices1 + weights1.unsqueeze(-1).repeat(1, self.num_agents-1) * all_opt_matrices2.repeat(1, self.num_agents-1)
            baseline_diversity_values2 = weights2 * all_baseline_matrices2.squeeze(1)
            opt_diversity_values2 = weights2 * all_opt_matrices2.squeeze(1)
        else:
            all_weights = weights1 + weights2
            if not self.conf_opt:
                baseline_diversity_values1 = torch.ones_like(all_weights).unsqueeze(-1).repeat(1, self.num_agents-1)*all_baseline_matrices2.repeat(1, self.num_agents-1) + (all_weights-1).unsqueeze(-1).repeat(1, self.num_agents-1) * all_baseline_matrices2.repeat(1, self.num_agents-1)
                opt_diversity_values1 = torch.ones_like(all_weights).unsqueeze(-1).repeat(1, self.num_agents-1)*all_opt_matrices2.repeat(1, self.num_agents-1) + (all_weights-1).unsqueeze(-1).repeat(1, self.num_agents-1) * all_opt_matrices2.repeat(1, self.num_agents-1)
            else:                                       
                baseline_diversity_values1 = torch.ones_like(all_weights).unsqueeze(-1).repeat(1, self.num_agents-1)*all_baseline_matrices1 + (all_weights-1).unsqueeze(-1).repeat(1, self.num_agents-1) * all_baseline_matrices2.repeat(1, self.num_agents-1)
                opt_diversity_values1 = torch.ones_like(all_weights).unsqueeze(-1).repeat(1, self.num_agents-1)*all_opt_matrices1 + (all_weights-1).unsqueeze(-1).repeat(1, self.num_agents-1) * all_opt_matrices2.repeat(1, self.num_agents-1)
            baseline_diversity_values2 = all_weights * all_baseline_matrices2.squeeze(1)
            opt_diversity_values2 = all_weights * all_opt_matrices2.squeeze(1)

        return opt_diversity_values1.detach() - baseline_diversity_values1.detach(), opt_diversity_values2.detach() - baseline_diversity_values2.detach(), all_entropy_weight1.detach(), all_entropy_weight2.detach()

    def compute_sp_actor_loss(self, obs, n_obs, acts, dones, trunc, rews, advantages1, advantages2, old_log_likelihoods, entropy_weight1, entropy_weight2):
        """
            A function that computes the policy's loss function based on self-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :param advantages: Weighted advantage value
            :param old_log_likelihoods: Previous policies log likelihood
            :param entropy_weight1: Variable entropy weights based on \alpha^1
            :param entropy_weight2: Variable entropy weights based on \alpha^2
            :return pol_loss: Policy loss
        """
        batch_size = obs.size()[0]
        obs_length = obs.size()[1]

        # Initialize empty lists that saves every important data for actor loss computation
        action_likelihood = []
        action_entropy = []

        anneal_end = self.config.train["anneal_end"] * self.projected_total_updates
        for idx in reversed(range(obs_length)):
            acts_idx = acts[:, idx, :, :]
            obs_idx = torch.cat([obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)

            action_logits = self.separate_act_select(obs_idx)
            action_distribution = dist.OneHotCategorical(logits=action_logits)

            # Append computed measures to their lists
            action_likelihood.append(action_distribution.log_prob(acts_idx))
            action_entropy.append(action_distribution.entropy())

        # Combine lists into a single tensor for actor loss from SP data
        action_entropies = torch.cat(action_entropy, dim=0)
        entropy_weights = torch.cat([
            entropy_weight1, entropy_weight2.unsqueeze(dim=-1)
        ], dim=-1)

        action_log_likelihoods = torch.cat(action_likelihood, dim=0)
        action_log_likelihoods1 = action_log_likelihoods[:,:-1]
        action_log_likelihoods2 = action_log_likelihoods[:,-1]
        entropy_loss = (entropy_weights * -action_entropies).sum(dim=-1).mean()

        old_log_likelihoods1 = old_log_likelihoods[:,:-1]
        old_log_likelihoods2 = old_log_likelihoods[:,-1]
        ratio1 = torch.exp(action_log_likelihoods1 - old_log_likelihoods1.detach())
        ratio2 = torch.exp(action_log_likelihoods2 - old_log_likelihoods2.detach())

        surr11 = ratio1 * advantages1
        surr21 = torch.clamp(
            ratio1,
            1 - self.config.train["eps_clip"],
            1 + self.config.train["eps_clip"]
        ) * advantages1
        
        surr12 = ratio2 * advantages2
        surr22 = torch.clamp(
            ratio2,
            1 - self.config.train["eps_clip"],
            1 + self.config.train["eps_clip"]
        ) * advantages2

        pol_list1 = torch.min(surr11, surr21)
        pol_list2 = torch.min(surr12, surr22)

        if self.config.train["with_dual_clip"]:
            pol_list1[advantages1 < 0] = torch.max(pol_list1[advantages1 < 0], self.config.train["dual_clip"]*advantages1[advantages1 < 0])
            pol_list2[advantages2 < 0] = torch.max(pol_list2[advantages2 < 0], self.config.train["dual_clip"]*advantages2[advantages2 < 0])
        
        all_pol = torch.cat([pol_list1, pol_list2.unsqueeze(-1)], dim=-1)
        pol_loss = -all_pol.mean()

        return pol_loss, entropy_loss

    def compute_sp_lagrange_loss(self, obs, n_obs, acts, dones, trunc, rews):
        """
            A function that computes the lagrange multiplier losses based on self-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return Lagrange multiplier loss
        """
        batch_size = obs.size()[0]
        obs_length = obs.size()[1]
        obs_only_length = self.effective_crit_obs_size - self.num_populations - self.num_agents

        # Initialize empty lists that saves every important data for actor loss computation
        diff_matrix1 = []
        saved_id11 = []
        saved_id21 = []

        if self.dual_constraints:
            diff_matrix2 = []
            saved_id12 = []
            saved_id22 = []

        # Initialize target diversity at end of episode to None
        target_diversity_value = None
        for idx in reversed(range(obs_length)):
            obs_idx = torch.cat([obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)
            n_obs_idx = torch.cat([n_obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)],
                                  dim=-1)
            sp_rl_rew = rews[:, idx, -1]
            sp_rl_done = dones[:, idx]
            sp_rl_trunc = trunc[:, idx]

            target_diversity_value_trunc = self.joint_action_value_functions[-1](
                n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1)
            )
            if idx == obs_length - 1:
                target_diversity_value = target_diversity_value_trunc

            offset = self.num_populations + self.num_agents
            if -offset + self.num_populations == 0:
                accessed_index = obs_idx[:, :, -offset:].argmax(dim=-1)
            else:
                accessed_index = obs_idx[:, :, -offset:-offset + self.num_populations].argmax(dim=-1)

            idx1, idx2 = accessed_index[:, 0], accessed_index[:, -1]
            repeated_idx1 = torch.repeat_interleave(idx1, self.num_populations, 0)
            repeated_idx1_final = torch.eye(self.num_populations).to(self.device)[repeated_idx1].unsqueeze(1).repeat(1, self.num_agents-1, 1)
            added_tensors1 = torch.eye(self.num_populations).to(self.device).repeat([batch_size,1]).unsqueeze(1)
            combined_tensors1 = torch.cat([repeated_idx1_final, added_tensors1], dim=1)

            if self.dual_constraints:
                repeated_idx2 = torch.repeat_interleave(idx2, self.num_populations, 0)
                repeated_idx2_final = torch.eye(self.num_populations).to(self.device)[repeated_idx2].unsqueeze(1)
                added_tensors2 = torch.eye(self.num_populations).to(self.device).repeat([batch_size,1]).unsqueeze(1).repeat(1, self.num_agents-1, 1) 
                combined_tensors2 = torch.cat([added_tensors2, repeated_idx2_final], dim=1)

            obs_only = obs_idx[:, :, :obs_only_length]
            r_obs_only = obs_only.repeat([1, self.num_populations, 1]).view(
                -1, obs_only.size()[-2], obs_only.size()[-1]
            )

            eval_input1 = torch.cat([r_obs_only, combined_tensors1, torch.eye(self.num_agents).repeat(r_obs_only.size()[0], 1, 1).to(self.device)], dim=-1)
            baseline_matrix1 = self.joint_action_value_functions[-1](
                eval_input1.view(eval_input1.size(0), 1, -1).squeeze(1)
            ).view(batch_size, self.num_populations)

            if self.dual_constraints:
                eval_input2 = torch.cat([r_obs_only, combined_tensors2, torch.eye(self.num_agents).repeat(r_obs_only.size()[0], 1, 1).to(self.device)], dim=-1)
                baseline_matrix2 = self.joint_action_value_functions[-1](
                    eval_input2.view(eval_input2.size(0), 1, -1).squeeze(1)
                ).view(batch_size, self.num_populations)

            target_diversity_value[sp_rl_trunc==1] = target_diversity_value_trunc[sp_rl_trunc==1]
            target_diversity_value = (
                    sp_rl_rew.view(-1, 1) + (
                        self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_diversity_value)
            ).detach()

            resized_target_values1 = target_diversity_value.repeat(1, self.num_populations)
            resized_target_values1[torch.arange(baseline_matrix1.size()[0]), idx1] = baseline_matrix1[torch.arange(baseline_matrix1.size()[0]), idx1] + self.config.train["tolerance_factor"]
            
            if self.dual_constraints:
                resized_target_values2 = target_diversity_value.repeat(1, self.num_populations)
                resized_target_values2[torch.arange(baseline_matrix2.size()[0]), idx2] = baseline_matrix2[torch.arange(baseline_matrix2.size()[0]), idx2] + self.config.train["tolerance_factor"]

            # Append computed measures to their lists
            diff_matrix1.append(resized_target_values1 - baseline_matrix1 - self.config.train["tolerance_factor"])
            saved_id11.append(torch.repeat_interleave(idx1, self.num_populations))
            saved_id21.append(torch.arange(self.num_populations).repeat(idx1.size()[0]).to(self.device))

            if self.dual_constraints:
                diff_matrix2.append(resized_target_values2 - baseline_matrix2 - self.config.train["tolerance_factor"])
                saved_id12.append(torch.repeat_interleave(idx2, self.num_populations))
                saved_id22.append(torch.arange(self.num_populations).repeat(idx2.size()[0]).to(self.device))

        # Combine lists into a single tensor for actor loss from SP data
        all_diff1 = torch.cat(diff_matrix1, dim=0).detach()
        all_id11 = torch.cat(saved_id11, dim=0)
        all_id21 = torch.cat(saved_id21, dim=0)

        if self.dual_constraints:
            all_diff2 = torch.cat(diff_matrix2, dim=0).detach()
            all_id12 = torch.cat(saved_id12, dim=0)
            all_id22 = torch.cat(saved_id22, dim=0)

            return (all_diff1, all_diff2, all_id11, all_id21, all_id12, all_id22), ((self.compute_const_lagrange1()**2).mean()**0.5), ((self.compute_const_lagrange2()**2).mean()**0.5)
        return (all_diff1, all_id11, all_id21), ((self.compute_const_lagrange1()**2).mean()**0.5)

        #TODO check

    def compute_xp_lagrange_loss(self, xp_obs, xp_acts, xp_rews, xp_dones, xp_trunc, xp_n_obses):
        """
            A function that computes the lagrange multiplier losses based on cross-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return Lagrange multiplier loss
        """
        batch_size = xp_obs.shape[0]
        obs_length = xp_obs.shape[1]
        obs_only_length = self.effective_crit_obs_size - self.num_populations - self.num_agents

        diff_matrix1 = []
        saved_id11 = []
        saved_id21 = []

        if self.dual_constraints:
            diff_matrix2 = []
            saved_id12 = []
            saved_id22 = []

        xp_targ_diversity_value = None
        for idx in reversed(range(obs_length)):
            xp_obs_id = torch.cat(
                [xp_obs[:, idx, :, :], torch.eye(self.num_agents).repeat([batch_size, 1, 1]).to(self.device)], dim=-1
            )

            xp_n_obs_inp = torch.cat([xp_n_obses[:, idx, :, :], torch.eye(self.num_agents).repeat(xp_obs_id.size()[0], 1, 1).to(self.device)], dim=-1)
            xp_targ_diversity_value_trunc = self.joint_action_value_functions[-1](
                xp_n_obs_inp.view(xp_n_obs_inp.size(0), 1, -1).squeeze(1)
            )

            if idx == obs_length - 1:
                xp_targ_diversity_value = xp_targ_diversity_value_trunc

            xp_rl_rew = xp_rews[:, idx, -1]
            xp_rl_done = xp_dones[:, idx]
            xp_rl_trunc = xp_trunc[:, idx]

            xp_obs_only = xp_obs_id[:, :, :obs_only_length]

            offset = self.num_populations + self.num_agents
            accessed_index = xp_obs_id[:, :, -offset:-offset + self.num_populations].argmax(dim=-1)

            idx1, idx2 = accessed_index[:, 0], accessed_index[:, -1]
            repeated_idx1 = torch.repeat_interleave(idx1, self.num_agents, 0)
            added_matrix1 = torch.eye(self.num_populations).to(self.device)[repeated_idx1].view(batch_size, -1, self.num_populations)
            xp_input1 = torch.cat([xp_obs_only, added_matrix1, torch.eye(self.num_agents).repeat(xp_obs_only.size()[0], 1, 1).to(self.device)], dim=-1)
            baseline_matrix1 = self.joint_action_value_functions[-1](
                xp_input1.view(xp_input1.size(0), 1, -1).squeeze(1),
            )

            if self.dual_constraints:
                repeated_idx2 = torch.repeat_interleave(idx2, self.num_agents, 0)
                added_matrix2 = torch.eye(self.num_populations).to(self.device)[repeated_idx2].view(batch_size, -1, self.num_populations)
                xp_input2 = torch.cat([xp_obs_only, added_matrix2, torch.eye(self.num_agents).repeat(xp_obs_only.size()[0], 1, 1).to(self.device)], dim=-1)
                baseline_matrix2 = self.joint_action_value_functions[-1](
                    xp_input2.view(xp_input2.size(0), 1, -1).squeeze(1),
                )

            xp_targ_diversity_value[xp_rl_trunc==1] = xp_targ_diversity_value_trunc[xp_rl_trunc==1] 
            xp_targ_diversity_value = (
                    xp_rl_rew.view(-1, 1) + (
                    self.config.train["gamma"] * (1 - xp_rl_done.view(-1, 1)) * xp_targ_diversity_value)
            ).detach()

            diff_matrix1.append(baseline_matrix1 - xp_targ_diversity_value - self.config.train["tolerance_factor"])
            saved_id11.append(idx1)
            saved_id21.append(idx2)

            if self.dual_constraints:
                diff_matrix2.append(baseline_matrix2 - xp_targ_diversity_value - self.config.train["tolerance_factor"])
                saved_id12.append(idx2)
                saved_id22.append(idx1)

        all_diff1 = torch.cat(diff_matrix1, dim=0).detach()
        all_id11 = torch.cat(saved_id11, dim=0)
        all_id21 = torch.cat(saved_id21, dim=0)

        if self.dual_constraints:
            all_diff2 = torch.cat(diff_matrix2, dim=0).detach()
            all_id12 = torch.cat(saved_id12, dim=0)
            all_id22 = torch.cat(saved_id22, dim=0)
        
        if self.dual_constraints:
            return (all_diff1, all_diff2, all_id11, all_id21, all_id12, all_id22), ((self.compute_const_lagrange1()**2).mean()**0.5), ((self.compute_const_lagrange2()**2).mean()**0.5)
        return (all_diff1, all_id11, all_id21), ((self.compute_const_lagrange1()**2).mean()**0.5)
        
    def compute_sp_critic_loss(
            self, obs_batch, n_obs_batch,
            acts_batch, sp_rew_batch, sp_done_batch, sp_truncs_batch
    ):
        """
            A function that computes critic loss based on self-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return Lagrange multiplier loss
        """
        batch_size = obs_batch.size()[0]
        obs_length = obs_batch.size()[1]

        predicted_values = []
        target_values = []
        all_target_values = None

        for idx in reversed(range(obs_length)):
            obs_idx = torch.cat([obs_batch[:,idx,:,:], torch.eye(self.num_agents).repeat(batch_size,1,1).to(self.device)], dim=-1)
            n_obs_idx = torch.cat([n_obs_batch[:,idx,:,:], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)

            futures = [
                torch.jit.fork(model, obs_idx.view(obs_idx.size(0), 1, -1).squeeze(1)) for model
                in self.joint_action_value_functions
            ]
            all_sp_v_values = [torch.jit.wait(fut) for fut in futures]
            all_sp_v_values = torch.cat(all_sp_v_values, dim=-1) 
            sp_rl_rew = sp_rew_batch[:, idx, :]
            sp_rl_done = sp_done_batch[:, idx]
            sp_rl_truncs = sp_truncs_batch[:, idx]

            futures = [
                torch.jit.fork(model, n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1)) for model
                in self.target_joint_action_value_functions
            ]
            all_target_values_trunc = [torch.jit.wait(fut) for fut in futures]
            all_target_values_trunc = torch.cat(all_target_values_trunc, dim=-1)

            if idx == obs_length-1:
                all_target_values = all_target_values_trunc
                
            all_target_values[sp_rl_truncs==1] = all_target_values_trunc[sp_rl_truncs==1]
            all_target_values = (
                sp_rl_rew + (self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1).repeat(1, self.num_agents)) * all_target_values)
            ).detach()

            predicted_values.append(all_sp_v_values)
            target_values.append(all_target_values.clone())

        predicted_values1 = torch.cat(predicted_values, dim=0)
        all_target_values1 = torch.cat(target_values, dim=0)
        
        sp_critic_loss1 = (0.5 * ((predicted_values1 - all_target_values1) ** 2)).mean()
        return sp_critic_loss1

    def compute_xp_advantages(self, xp_obs, xp_acts, xp_rews, xp_dones, xp_truncs, xp_n_obses):
        """
            A function that computes the weighted advantage function as described in Expression 14 based on XP interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return opt_diversity_values.detach() - baseline_diversity_values.detach(): weighted advantage function
            :return all_entropy_weight1.detach(): Variable entropy weights given \alpha^1
            :return all_entropy_weight1.detach(): Variable entropy weights given \alpha^2
        """
        batch_size = xp_obs.size()[0]
        obs_length = xp_obs.size()[1]

        opt_xp_matrics = []
        baseline_xp_matrics = []
        lagrangian_weights1 = []
        if self.dual_constraints:
            lagrangian_weights2 = []
        entropy_weight1 = []
        entropy_weight2 = []

        xp_targ_diversity_value = None
        # Compute added stuff related to index

        lagrange_matrix_mean_norm1 = self.compute_const_lagrange1().mean(dim=-1, keepdim=False)
        lagrange_matrix_mean_norm2 = self.compute_const_lagrange1().mean(dim=0, keepdim=False)

        if self.dual_constraints:
            lagrange_matrix_mean_norm1 = self.compute_const_lagrange1().mean(dim=-1, keepdim=False)
            lagrange_matrix_mean_norm2 = self.compute_const_lagrange2().mean(dim=-1, keepdim=False)

        pos_entropy_weights1 = (lagrange_matrix_mean_norm1 / self.normalizer1) * self.config.loss_weights[
            "entropy_regularizer_loss"
        ]
        pos_entropy_weights2 = (lagrange_matrix_mean_norm2 / self.normalizer2) * self.config.loss_weights[
            "entropy_regularizer_loss"
        ]

        for idx in reversed(range(obs_length)):
            xp_obs_id = torch.cat([xp_obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)
            xp_n_obses_id = torch.cat([xp_n_obses[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)
            xp_targ_diversity_trunc = self.joint_action_value_functions[-1](
                xp_n_obses_id.view(xp_n_obses_id.size(0), 1, -1).squeeze(1)
            )

            if idx == obs_length - 1:
                xp_targ_diversity_value = xp_targ_diversity_trunc

            xp_rl_rew = xp_rews[:, idx, -1]
            xp_rl_done = xp_dones[:, idx]
            xp_rl_truncs = xp_truncs[:, idx]

            offset = self.num_populations + self.num_agents
            accessed_index = xp_obs_id[:, :, -offset:-offset + self.num_populations].argmax(dim=-1)

            idx1, idx2 = accessed_index[:, 0], accessed_index[:, -1]
            baseline_matrix = self.joint_action_value_functions[-1](
                xp_obs_id.view(xp_obs_id.size(0), 1, -1).squeeze(1)
            )

            lagrangian_matrix1 = self.compute_const_lagrange1().unsqueeze(0).repeat(batch_size, 1, 1)
            if self.dual_constraints:  
                lagrangian_matrix2 = self.compute_const_lagrange2().unsqueeze(0).repeat(batch_size, 1, 1)

            xp_targ_diversity_value[xp_rl_truncs==1] = xp_targ_diversity_trunc[xp_rl_truncs==1]
            xp_targ_diversity_value = (
                    xp_rl_rew.view(-1, 1) + (
                    self.config.train["gamma"] * (1 - xp_rl_done.view(-1, 1)) * xp_targ_diversity_value)
            ).detach()

            lagrangian_weights1.append(lagrangian_matrix1[torch.arange(batch_size), idx1, idx2].unsqueeze(-1))
            if self.dual_constraints:
                lagrangian_weights2.append(lagrangian_matrix2[torch.arange(batch_size), idx2, idx1].unsqueeze(-1))
            opt_xp_matrics.append(-xp_targ_diversity_value.clone())
            baseline_xp_matrics.append(-baseline_matrix)
            entropy_weight1.append(pos_entropy_weights1[idx1])
            entropy_weight2.append(pos_entropy_weights2[idx2])

        all_baseline_matrices = torch.cat(baseline_xp_matrics, dim=0)
        all_opt_matrices = torch.cat(opt_xp_matrics, dim=0)
        if self.dual_constraints:   
            all_lagrangian_matrices2 = torch.cat(lagrangian_weights2, dim=0)
        all_lagrangian_matrices1 = torch.cat(lagrangian_weights1, dim=0)
        all_entropy_weights1 = torch.cat(entropy_weight1, dim=0)
        all_entropy_weights2 = torch.cat(entropy_weight2, dim=0)

        if not self.dual_constraints:
            xp_lagrange_advantages = ((all_opt_matrices-all_baseline_matrices) * (
                    all_lagrangian_matrices1
                )
            ).squeeze(1)
        else:
            xp_lagrange_advantages = ((all_opt_matrices-all_baseline_matrices) * (
                    all_lagrangian_matrices1 + all_lagrangian_matrices2
                )
            ).squeeze(1)

        return xp_lagrange_advantages.detach(), all_entropy_weights1.detach(), all_entropy_weights2.detach()

    def compute_xp_old_probs(self, xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses):
        """
            A function that computes the previous policy's (before update) probability of selecting an action based on cross-play interaction data.
            Required for MAPPO update.
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return old_log_likelihoods: Previous policies' action log probability
        """
        obs_length = xp_obs.shape[1]
        action_likelihood = []

        for idx in reversed(range(obs_length)):
            xp_obs_id = xp_obs[:, idx, :, :]
            xp_acts_idx = xp_acts[:, idx, :, :]
            xp_act_log_input = torch.cat(
                [xp_obs_id, torch.eye(self.num_agents).repeat(xp_obs_id.size()[0], 1, 1).to(self.device)],
                dim=-1
            )

            action_logits = self.separate_act_select_old(xp_act_log_input)
            final_selected_logits = action_logits
            action_distribution = dist.OneHotCategorical(logits=final_selected_logits)
            action_likelihood.append(action_distribution.log_prob(xp_acts_idx))

        action_log_likelihoods = torch.cat(action_likelihood, dim=0)
        return action_log_likelihoods

    def compute_xp_actor_loss(self, xp_obs, xp_acts, xp_rews, xp_dones, xp_trunc, xp_n_obses, advantages, old_log_likelihoods, entropy_weight1, entropy_weight2):
        """
            A function that computes the policy's loss function based on cross-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :param advantages: Weighted advantage value
            :param old_log_likelihoods: Previous policies log likelihood
            :param entropy_weight1: Variable entropy weights based on \alpha^1
            :param entropy_weight2: Variable entropy weights based on \alpha^2
            :return pol_loss: Policy loss
        """
        obs_length = xp_obs.shape[1]
        action_likelihood = []
        action_entropies = []

        for idx in reversed(range(obs_length)):
            xp_obs_id = xp_obs[:, idx, :, :]
            xp_acts_idx = xp_acts[:, idx, :, :]
            xp_act_log_input = torch.cat(
                [xp_obs_id, torch.eye(self.num_agents).repeat(xp_obs_id.size()[0], 1, 1).to(self.device)],
                dim=-1
            )

            action_logits = self.separate_act_select(xp_act_log_input)
            action_distribution = dist.OneHotCategorical(logits=action_logits)
            action_likelihood.append(action_distribution.log_prob(xp_acts_idx))
            action_entropies.append(action_distribution.entropy())

        action_log_likelihoods = torch.cat(action_likelihood, dim=0)
        action_entropies = torch.cat(action_entropies, dim=0)

        entropy_weights = torch.cat([entropy_weight1.unsqueeze(dim=-1).repeat(1, self.num_agents-1), entropy_weight2.unsqueeze(dim=-1)], dim=-1)
        entropy_loss = (entropy_weights * -action_entropies).sum(dim=-1).mean()
        xp_ratio = torch.exp(action_log_likelihoods - old_log_likelihoods.detach())
        repeated_advantages = advantages.unsqueeze(-1).repeat(1, self.num_agents)
        surr1 = xp_ratio * repeated_advantages
        surr2 = torch.clamp(
            xp_ratio,
            1 - self.config.train["eps_clip"],
            1 + self.config.train["eps_clip"]
        ) * repeated_advantages

        xp_pol_list = torch.min(surr1, surr2)
        if self.config.train["with_dual_clip"]:
            xp_pol_list[repeated_advantages < 0] = torch.max(xp_pol_list[repeated_advantages < 0], self.config.train["dual_clip"]*repeated_advantages[repeated_advantages < 0])
        
        xp_pol_loss = -xp_pol_list.mean()

        return xp_pol_loss, entropy_loss

    def compute_xp_critic_loss(self, xp_obs, xp_acts, xp_rew, xp_dones, xp_truncs, xp_n_obses):
        """
            A function that computes critic loss based on cross-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return Lagrange multiplier loss
        """
        batch_size = xp_obs.shape[0]
        obs_length = xp_obs.shape[1]

        predicted_values = []
        target_values = []

        all_targ_values = None

        for idx in reversed(range(obs_length)):
            xp_obs_id = xp_obs[:, idx, :, :]
            xp_critic_state_input = torch.cat(
                [xp_obs_id, torch.eye(self.num_agents).repeat([batch_size, 1, 1]).to(self.device)],dim=-1
            )

            futures = [
                torch.jit.fork(model, xp_critic_state_input.view(xp_critic_state_input.size(0), 1, -1).squeeze(1)) for model
                in self.joint_action_value_functions
            ]
            xp_v_values = [torch.jit.wait(fut) for fut in futures]
            xp_v_values = torch.cat(xp_v_values, dim=-1)

            xp_n_obses_id = xp_n_obses[:, idx, :, :]
            xp_critic_n_state_input = torch.cat(
                [xp_n_obses_id, torch.eye(self.num_agents).repeat([batch_size, 1, 1]).to(self.device)],
                dim=-1
            )

            futures = [
                torch.jit.fork(model, xp_critic_n_state_input.view(xp_critic_n_state_input.size(0), 1, -1).squeeze(1)) for model
                in self.target_joint_action_value_functions
            ]
            all_targ_values_trunc = [torch.jit.wait(fut) for fut in futures]
            all_targ_values_trunc = torch.cat(all_targ_values_trunc, dim=-1)

            if idx == obs_length - 1:
                all_targ_values = all_targ_values_trunc

            xp_rl_rew = xp_rew[:, idx, :]
            xp_rl_done = xp_dones[:, idx]
            xp_rl_truncs = xp_truncs[:, idx]

            all_targ_values[xp_rl_truncs==1] = all_targ_values_trunc[xp_rl_truncs==1]
            all_targ_values = (
                xp_rl_rew + (self.config.train["gamma"] * (1 - xp_rl_done.view(-1, 1).repeat(1, self.num_agents)) * all_targ_values)
            ).detach()

            predicted_values.append(xp_v_values)
            target_values.append(all_targ_values.clone())

        predicted_values = torch.cat(predicted_values, dim=0)
        target_values_combined = torch.cat(target_values, dim=0)
        xp_critic_loss1 = (0.5 * ((predicted_values - target_values_combined) ** 2)).mean()
        return xp_critic_loss1


    def update(self, batches, xp_batches):
        """
            A method that updates the joint policy model following sampled self-play and cross-play experiences.
            :param batches: A batch of obses and acts sampled from self-play experience replay.
            :param xp_batches: A batch of experience from cross-play experience replay.
        """

        self.total_updates += 1

        # Get obs and acts batch and prepare inputs to model.
        obs_batch, acts_batch = torch.tensor(batches[0]).to(self.device), torch.tensor(batches[1]).to(self.device)
        sp_n_obs_batch = torch.tensor(batches[2]).to(self.device)
        sp_done_batch = torch.tensor(batches[3]).double().to(self.device)
        sp_trunc_batch = torch.tensor(batches[4]).double().to(self.device)
        rewards_batch = torch.tensor(batches[5]).double().to(self.device)

        xp_obs, xp_acts = torch.tensor(xp_batches[0]).to(self.device), torch.tensor(xp_batches[1]).to(self.device)
        xp_n_obses = torch.tensor(xp_batches[2]).to(self.device)
        xp_dones = torch.tensor(xp_batches[3]).double().to(self.device)
        xp_trunc_batch = torch.tensor(xp_batches[4]).double().to(self.device)
        xp_rews = torch.tensor(xp_batches[5]).double().to(self.device)

        sp_advantages1, sp_advantages2, entropy_weight1, entropy_weight2 = self.compute_sp_advantages(
            obs_batch, sp_n_obs_batch, acts_batch, sp_done_batch, sp_trunc_batch, rewards_batch
        )

        for pol, old_pol in zip(self.joint_policy, self.old_joint_policy):
            for target_param, param in zip(old_pol.parameters(), pol.parameters()):
                target_param.data.copy_(param.data)

        sp_old_log_probs = self.compute_sp_old_probs(
            obs_batch, acts_batch
        )

        xp_advantages, xp_ent_weight1,  xp_ent_weight2 = self.compute_xp_advantages(
            xp_obs, xp_acts, xp_rews, xp_dones, xp_trunc_batch, xp_n_obses
        )

        xp_old_log_probs = self.compute_xp_old_probs(
            xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses
        )

        for _ in range(self.config.train["epochs_per_update"]):
            self.actor_optimizer.zero_grad()

            # Compute SP Actor Loss
            sp_pol_loss, sp_action_entropies = self.compute_sp_actor_loss(
                obs_batch, sp_n_obs_batch, acts_batch, sp_done_batch, sp_trunc_batch, rewards_batch, sp_advantages1, sp_advantages2, sp_old_log_probs, entropy_weight1, entropy_weight2
            )

            # Compute XP Actor Loss
            total_xp_actor_loss, total_xp_entropy_loss = self.compute_xp_actor_loss(
                xp_obs, xp_acts, xp_rews, xp_dones, xp_trunc_batch, xp_n_obses, xp_advantages, xp_old_log_probs, xp_ent_weight1,  xp_ent_weight2
            )

            xp_multiplier = (self.config.env.parallel["xp_collection"]+0.0)/self.config.env.parallel["sp_collection"]
            total_actor_loss = sp_pol_loss + xp_multiplier*total_xp_actor_loss + sp_action_entropies + xp_multiplier*total_xp_entropy_loss
            total_actor_loss.backward()

            if self.config.train['max_grad_norm'] > 0:
                for model in self.joint_policy:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.train['max_grad_norm'])

            self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        # Compute SP Critic Loss
        sp_critic_loss = self.compute_sp_critic_loss(
            obs_batch, sp_n_obs_batch, acts_batch, rewards_batch, sp_done_batch, sp_trunc_batch
        )

        # Compute SP Lagrange Loss
        if (not self.constant_lagrange) and self.total_updates % self.config.train["lagrange_update_period"] == 0:
            if not self.dual_constraints:
                sp_lagrange_data, lagrange_mult_norm_sp1 = self.compute_sp_lagrange_loss(
                    obs_batch, sp_n_obs_batch, acts_batch, sp_done_batch, sp_trunc_batch, rewards_batch
                )
            else:
                sp_lagrange_data, lagrange_mult_norm_sp1, lagrange_mult_norm_sp2 = self.compute_sp_lagrange_loss(
                    obs_batch, sp_n_obs_batch, acts_batch, sp_done_batch, sp_trunc_batch, rewards_batch
                )

        # Get XP data and preprocess it for matrix computation
        # total_xp_critic_loss, total_xp_actor_loss, total_xp_entropy_loss, xp_pred_div, xp_lagrange_loss = 0, 0, 0, 0, 0
        
        # Compute XP Critic Loss
        total_xp_critic_loss = self.compute_xp_critic_loss(
            xp_obs, xp_acts, xp_rews, xp_dones, xp_trunc_batch, xp_n_obses
        )

        # Compute XP Lagrange Loss
        if (not self.constant_lagrange) and self.total_updates % self.config.train["lagrange_update_period"] == 0:
            if not self.dual_constraints:
                xp_lagrange_data, lagrange_mult_norm_xp1 = self.compute_xp_lagrange_loss(
                    xp_obs, xp_acts, xp_rews, xp_dones, xp_trunc_batch, xp_n_obses
                )
            else: 
                xp_lagrange_data, lagrange_mult_norm_xp1, lagrange_mult_norm_xp2 = self.compute_xp_lagrange_loss(
                    xp_obs, xp_acts, xp_rews, xp_dones, xp_trunc_batch, xp_n_obses
                )

        total_critic_loss = sp_critic_loss * self.config.loss_weights["sp_val_loss_weight"] + total_xp_critic_loss * self.config.loss_weights["xp_val_loss_weight"]
        if (not self.constant_lagrange) and self.total_updates % self.config.train["lagrange_update_period"] == 0:
            #total_lagrange_loss = self.config.loss_weights["lagrange_weights"] * (sp_lagrange_loss + xp_lagrange_loss)
            if not self.dual_constraints:   
                sp_all_diff1, sp_all_id11, sp_all_id21 = sp_lagrange_data
                xp_all_diff1, xp_all_id11, xp_all_id21 = xp_lagrange_data

                all_diff1 = torch.cat([sp_all_diff1.view(-1), xp_all_diff1.view(-1)], dim=0)
                all_id11 = torch.cat([sp_all_id11.view(-1), xp_all_id11.view(-1)], dim=0)
                all_id21 = torch.cat([sp_all_id21.view(-1), xp_all_id21.view(-1)], dim=0)

                for ii in range(self.num_populations):
                    for jj in range(self.num_populations):
                        if ii != jj:
                            eligible_diffs = all_diff1[torch.logical_and(all_id11 == ii, all_id21 == jj)]
                            if eligible_diffs.size()[0] != 0:
                                self.lagrange_multiplier_matrix1[ii][jj] = F.relu(F.relu(self.lagrange_multiplier_matrix1[ii][jj]) - (self.config.train["lagrange_lr"] * self.config.loss_weights["lagrange_weights"] * eligible_diffs.mean()))
            else:
                sp_all_diff1, sp_all_diff2, sp_all_id11, sp_all_id21, sp_all_id12, sp_all_id22 = sp_lagrange_data
                xp_all_diff1, xp_all_diff2, xp_all_id11, xp_all_id21, xp_all_id12, xp_all_id22 = xp_lagrange_data

                all_diff1 = torch.cat([sp_all_diff1.view(-1), xp_all_diff1.view(-1)], dim=0)
                all_id11 = torch.cat([sp_all_id11.view(-1), xp_all_id11.view(-1)], dim=0)
                all_id21 = torch.cat([sp_all_id21.view(-1), xp_all_id21.view(-1)], dim=0)
                all_id12 = torch.cat([sp_all_id12.view(-1), xp_all_id12.view(-1)], dim=0)
                all_id22 = torch.cat([sp_all_id22.view(-1), xp_all_id22.view(-1)], dim=0)
                all_diff2 = torch.cat([sp_all_diff2.view(-1), xp_all_diff2.view(-1)], dim=0)
                for ii in range(self.num_populations):
                    for jj in range(self.num_populations):
                        if ii != jj:
                            eligible_diffs = all_diff1[torch.logical_and(all_id11 == ii, all_id21 == jj)]
                            if eligible_diffs.size()[0] != 0:
                                self.lagrange_multiplier_matrix1[ii][jj] = F.relu(F.relu(self.lagrange_multiplier_matrix1[ii][jj]) - (self.config.train["lagrange_lr"] * self.config.loss_weights["lagrange_weights"] * eligible_diffs.mean()))
                            eligible_diffs2 = all_diff2[torch.logical_and(all_id12 == ii, all_id22 == jj)]
                            if eligible_diffs2.size()[0] != 0:
                                self.lagrange_multiplier_matrix2[ii][jj] = F.relu(F.relu(self.lagrange_multiplier_matrix2[ii][jj]) - (self.config.train["lagrange_lr"] * self.config.loss_weights["lagrange_weights"] * eligible_diffs2.mean()))
                 
        # Write losses to logs
        self.next_log_update += self.logger.train_log_period
        train_step = (self.total_updates-1) * self.logger.steps_per_update
        self.logger.log_item("Train/sp/actor_loss", sp_pol_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/sp/critic_loss", sp_critic_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/sp/entropy", sp_action_entropies,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/xp/actor_loss", total_xp_actor_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/xp/critic_loss", total_xp_critic_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/xp/entropy", total_xp_entropy_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        if self.total_updates % self.config.train["lagrange_update_period"] == 0:
            if not self.dual_constraints:
                self.logger.log_item("Train/sp/lagrange_mult_norm", lagrange_mult_norm_sp1,
                                        train_step=train_step, updates=self.total_updates-1)
                self.logger.log_item("Train/xp/lagrange_mult_norm", lagrange_mult_norm_xp1,
                                        train_step=train_step, updates=self.total_updates-1)
            else:
                self.logger.log_item("Train/sp/lagrange_mult_norm", lagrange_mult_norm_sp1+lagrange_mult_norm_sp2,
                                        train_step=train_step, updates=self.total_updates-1)
                self.logger.log_item("Train/xp/lagrange_mult_norm", lagrange_mult_norm_xp1+lagrange_mult_norm_xp2,
                                        train_step=train_step, updates=self.total_updates-1)
        
        self.logger.commit()
        total_critic_loss.backward()

        # Clip grads if necessary
        if self.config.train['max_grad_norm'] > 0:
            for idx, model in enumerate(self.joint_policy):
                nn.utils.clip_grad_norm_(model.parameters(), self.config.train['max_grad_norm'])

        # Log grad magnitudes if specified by config.
        if self.config.logger["log_grad"]:
            for idx, model in enumerate(self.joint_policy):
                for name, param in model.named_parameters():
                    if not param.grad is None:
                        self.logger.log_item(
                            f"Train/grad/actor_{idx}_{name}",
                            torch.abs(param.grad).mean(),
                            train_step=self.total_updates-1
                        )

            for idx, model in enumerate(self.joint_action_value_functions):
                for name, param in model.named_parameters():
                    if not param.grad is None:
                        self.logger.log_item(
                            f"Train/grad/critic_{name}",
                            torch.abs(param.grad).mean(),
                            train_step=self.total_updates-1
                            )

        self.critic_optimizer.step()
        self.soft_copy(self.config.train["target_update_rate"])

    def save_model(self, int_id, save_model=False):
        if not save_model:
            return
        torch.save(self.lagrange_multiplier_matrix1,
                   f"models/model_{int_id}-lagrange1.pt")
        super().save_model(int_id, save_model)

    def load_model(self, int_id, overridden_model_dir=None):
        model_dir = self.config.env['model_load_dir']
        if self.mode == "train":
            self.lagrange_multiplier_matrix1.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-lagrange1.pt")
            )
        super().load_model(int_id, overridden_model_dir)

