from Network import fc_network
from torch import optim, nn
from torch.autograd import Variable
import numpy as np
import torch.distributions as dist
import torch
import torch.nn.functional as F
import math

class MAPPOAgentPopulations(object):
    """
        A class that encapsulates the joint policy model controlling each agent population.
    """
    def __init__(self, obs_size, num_agents, num_populations, configs, act_sizes, device, logger, mode="train"):
        """
            :param obs_size: The length of the entire observation inputted to the model (already includes appended population id)
            :param agent_o_size: The length of u (graph-level features) for the model
            :param num_agents: Number of agents in env.
            :param num_populations: Number of trained populations.
            :param configs: The dictionary config containing hyperparameters of the model
            :param act_sizes: Per-agent action space dimension
            :param device: Device to store model
            :param loss_writer: Tensorboard writer to log results
            :param model_grad_writer: Tensorboard writer to log model gradient magnitude
        """
        # Observation size (but with one-hot population_data)
        self.obs_size = obs_size
        self.config = configs
        self.act_size = act_sizes
        self.device = device
        self.total_updates = 0
        self.next_log_update = 0
        self.num_agents = num_agents
        self.mode = mode
        self.id_length = self.config.populations["num_populations"]
        self.num_populations = num_populations
        self.effective_crit_obs_size = obs_size + self.num_agents

        total_checkpoints = self.config.run["total_checkpoints"]
        if not "sp_collection" in self.config.env.parallel.keys() and not "xp_collection" in self.config.env.parallel.keys():
            timesteps_per_checkpoint = self.config.run["num_timesteps"]//(total_checkpoints*(self.config.env.parallel["agent1_collection"]+self.config.env.parallel["agent2_collection"]))
        else:
            timesteps_per_checkpoint = self.config.run["num_timesteps"]//(total_checkpoints*(self.config.env.parallel["sp_collection"]+self.config.env.parallel["xp_collection"]))
        self.projected_total_updates = total_checkpoints * math.ceil((timesteps_per_checkpoint+0.0)/self.config.train["timesteps_per_update"])

        actor_dims = [self.obs_size-self.num_populations, *self.config.model.actor_dims, self.act_size]
        critic_dims = [self.num_agents * (self.obs_size-self.num_populations), *self.config.model.critic_dims, 1]

        init_ortho = self.config.model.get("init_ortho", True)

        if self.num_populations == 0:
            self.num_populations += 1

        self.joint_policy = [
            fc_network(actor_dims, init_ortho).double().to(self.device)
            for _ in range(self.num_agents*self.num_populations)
        ]

        self.old_joint_policy = [
            fc_network(actor_dims, init_ortho).double().to(self.device)
            for _ in range(self.num_agents*self.num_populations)
        ]

        for pol, old_pol in zip(self.joint_policy, self.old_joint_policy):
            for target_param, param in zip(old_pol.parameters(), pol.parameters()):
                target_param.data.copy_(param.data)

        # Collection of centralized critics. Each trained policy has one.
        self.joint_action_value_functions = [fc_network(critic_dims, init_ortho).double().to(self.device) for _ in range(self.num_agents*self.num_populations)]
        self.target_joint_action_value_functions = [fc_network(critic_dims, init_ortho).double().to(self.device) for _ in range(self.num_agents*self.num_populations)]
        self.hard_copy()

        # Initialize optimizer
        params_list = None
        critic_params_list = [
            *(param for critic in self.joint_action_value_functions for param in critic.parameters())
        ]

        actors_params_list = [
            *(param for actor in self.joint_policy for param in actor.parameters())
        ]

        self.critic_optimizer = optim.Adam(
            critic_params_list,
            lr=self.config.train["lr"]
        )

        self.actor_optimizer = optim.Adam(
            actors_params_list,
            lr=self.config.train["lr"]
        )

        self.logger = logger

    def to_one_hot(self, actions):
        """
            A method that changes agents' actions into a one-hot encoding format.
            :param actions: Agents' actions in form of an integer.
            :return: Agents' actions in a one-hot encoding form.
        """
        act_indices = np.asarray(actions).astype(int)
        one_hot_acts = np.eye(self.act_size)[act_indices]
        return one_hot_acts
    
    def onehot_from_logits(self, logits):
        """
        Method to change action logits into actions under a one hot format.
        :param logits: Action logits
        :return: argmax_acs: Action under one-hot format
        """
        argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).double()
        return argmax_acs

    def separate_act_select(self, input):
        """
        A method that quickly computes all policies' logits even if they have separate parameters via jit.
        It works by splitting the input vector based on the ID of the population that handles the input and the agent ID.
        These vectors are appended at the end of input.
        :param input: Input vector (Format: Obs + population one-hot encoding + agent id one-hot encoding)
        :return: logits: Action logits
        """
        additional_input_length = self.num_agents
        per_id_input = [None] * (self.num_agents * self.num_populations)
        for pop_id in range(self.num_populations):
            for a_id in range(self.num_agents):
                per_id_input[self.num_agents*pop_id+a_id] = input[
                    torch.logical_and(
                        input[:, :, -(self.num_populations + additional_input_length) + pop_id] == 1,
                        input[:, :, -self.num_agents + a_id] == 1
                    )
                ][:, :-(self.num_populations+additional_input_length)]

        per_id_input_filtered = [(idx,inp) for idx, inp in enumerate(per_id_input) if not inp.nelement() == 0]
        executed_models = [policy for idx, policy in enumerate(self.joint_policy) if
                           not per_id_input[idx].nelement() == 0]

        futures = [
            torch.jit.fork(model, per_id_input_filtered[i][1]) for i, model
            in enumerate(executed_models)
        ]

        results = [torch.jit.wait(fut) for fut in futures]
        logits = torch.zeros([input.size()[0], input.size()[1], self.act_size]).double().to(self.device)

        # Rearrange output into two 2-D vectors of size (Batch size x Num actions)
        id = 0
        for idx, _ in per_id_input_filtered:
            population_id = idx // self.num_agents
            agent_id = idx % self.num_agents
            logits[
                torch.logical_and(
                    input[:, :, -(self.num_populations + additional_input_length) + population_id] == 1,
                    input[:, :, -self.num_agents + agent_id] == 1
                )
            ] = results[id]
            id += 1

        return logits

    def separate_act_select_old(self, input):
        """
        A method that quickly computes all old policies' logits even if they have separate parameters via jit.
        It works by splitting the input vector based on the ID of the population that handles the input and the agent ID.
        These vectors are appended at the end of input.
        :param input: Input vector (Format: Obs + population one-hot encoding + agent id one-hot encoding)
        :return: logits: Action logits
        """
        additional_input_length = self.num_agents
        per_id_input = [None] * (self.num_agents * self.num_populations)
        for pop_id in range(self.num_populations):
            for a_id in range(self.num_agents):
                per_id_input[self.num_agents*pop_id+a_id] = input[
                    torch.logical_and(
                        input[:, :, -(self.num_populations + additional_input_length) + pop_id] == 1,
                        input[:, :, -self.num_agents + a_id] == 1
                    )
                ][:, :-(self.num_populations+additional_input_length)]

        per_id_input_filtered = [(idx,inp) for idx, inp in enumerate(per_id_input) if not inp.nelement() == 0]
        executed_models = [policy for idx, policy in enumerate(self.old_joint_policy) if
                           not per_id_input[idx].nelement() == 0]

        futures = [
            torch.jit.fork(model, per_id_input_filtered[i][1]) for i, model
            in enumerate(executed_models)
        ]

        results = [torch.jit.wait(fut) for fut in futures]
        logits = torch.zeros([input.size()[0], input.size()[1], self.act_size]).double().to(self.device)

        # Rearrange output into two 2-D vectors of size (Batch size x Num ACTIONS)
        id = 0
        for idx, _ in per_id_input_filtered:
            population_id = idx // self.num_agents
            agent_id = idx % self.num_agents
            logits[
                torch.logical_and(
                    input[:, :, -(self.num_populations + additional_input_length) + population_id] == 1,
                    input[:, :, -self.num_agents + agent_id] == 1
                )
            ] = results[id]
            id += 1

        return logits
    
    def separate_critic_eval(self,input):
        """
        A method that quickly computes all critics' estimated values even if they have separate parameters via jit.
        It works by splitting the input vector based on the ID of the population that handles the input.
        These vectors are appended at the end of input.
        :param input: Input vector (Format: Obs + population one-hot encoding )
        :return: logits: Value function estimate
        """
        per_id_input = [None] * (2 * self.num_populations)
        input1, input2 = input[:,0,:], input[:,1,:]
        
        for pop_id in range(self.num_populations):
            agent1_features = input1[
                input1[:, -self.num_populations + pop_id] == 1
            ][:, :-self.num_populations]
            agent2_features = input2[
                input2[:, -self.num_populations + pop_id] == 1
            ][:, :-self.num_populations]
            per_id_input[2*pop_id] = torch.cat([agent1_features, agent2_features], dim=-1)
            per_id_input[2*pop_id+1] = torch.cat([agent1_features, agent2_features], dim=-1)

        per_id_input_filtered = [(idx,inp) for idx, inp in enumerate(per_id_input) if not inp.nelement() == 0]
        executed_models = [critic for idx, critic in enumerate(self.joint_action_value_functions) if
                           not per_id_input[idx].nelement() == 0]

        futures = [
            torch.jit.fork(model, per_id_input_filtered[i][1]) for i, model
            in enumerate(executed_models)
        ]

        results = [torch.jit.wait(fut) for fut in futures]

        #Get default value functions for Agent 1 and 2
        vals1 = torch.zeros([input.size()[0], 1]).double().to(self.device)
        vals2 = torch.zeros([input.size()[0], 1]).double().to(self.device)

        # Rearrange output into two 2-D vectors of size (Batch size x 1)
        id = 0
        for idx, _ in per_id_input_filtered:
            population_id = idx // 2
            if idx % 2 == 0:
                vals1[
                    input1[:, -self.num_populations + population_id] == 1
                ] = results[id]
                id += 1
            else:
                vals2[
                    input2[:, -self.num_populations + population_id] == 1
                ] = results[id]
                id += 1

        return vals1, vals2
    
    def separate_target_critic_eval(self,input):
        """
        A method that quickly computes all target critics' estimated values even if they have separate parameters via jit.
        It works by splitting the input vector based on the ID of the population that handles the input.
        These vectors are appended at the end of input.
        :param input: Input vector (Format: Obs + population one-hot encoding )
        :return: logits: Value function estimate
        """

        per_id_input = [None] * (2 * self.num_populations)
        input1, input2 = input[:,0,:], input[:,1,:]

        for pop_id in range(self.num_populations):
            agent1_features = input1[
                input1[:, -self.num_populations + pop_id] == 1
            ][:, :-self.num_populations]
            agent2_features = input2[
                input2[:, -self.num_populations + pop_id] == 1
            ][:, :-self.num_populations]
            per_id_input[2*pop_id] = torch.cat([agent1_features, agent2_features], dim=-1)
            per_id_input[2*pop_id+1] = torch.cat([agent1_features, agent2_features], dim=-1)

        per_id_input_filtered = [(idx,inp) for idx, inp in enumerate(per_id_input) if not inp.nelement() == 0]
        executed_models = [critic for idx, critic in enumerate(self.target_joint_action_value_functions) if
                           not per_id_input[idx].nelement() == 0]

        futures = [
            torch.jit.fork(model, per_id_input_filtered[i][1]) for i, model
            in enumerate(executed_models)
        ]

        results = [torch.jit.wait(fut) for fut in futures]

        #Get default value functions for Agent 1 and 2
        vals1 = torch.zeros([input.size()[0], 1]).double().to(self.device)
        vals2 = torch.zeros([input.size()[0], 1]).double().to(self.device)

        # Rearrange output into two 2-D vectors of size (Batch size x 1)
        id = 0
        for idx, _ in per_id_input_filtered:
            population_id = idx // self.num_agents
            if idx % 2 == 0:
                vals1[
                    input1[:, -self.num_populations + population_id] == 1
                ] = results[id]
                id += 1
            else:
                vals2[
                    input2[:, -self.num_populations + population_id] == 1
                ] = results[id]
                id += 1

        return vals1, vals2

    def decide_acts(self, obs_w_commands, with_log_probs=False, eval=False, epsilon=None):
        """
            A method to decide the actions of agents given obs & target returns.
            :param obs_w_commands: A numpy array that has the obs concatenated with the target returns.
            :return: Sampled actions under the specific obs.
        """
        obs_w_commands = torch.tensor(obs_w_commands).to(self.device)
        batch_size, num_agents = obs_w_commands.size()[0], obs_w_commands.size()[1]

        in_population_agent_id = torch.eye(num_agents).repeat(batch_size, 1, 1).to(self.device)
        obs_w_commands = torch.cat([obs_w_commands, in_population_agent_id], dim=-1)

        act_logits = self.separate_act_select(obs_w_commands)
        if not eval:
            particle_dist = dist.OneHotCategorical(logits=act_logits)
            original_acts = particle_dist.sample()
            acts = original_acts.argmax(dim=-1)
        else:
            particle_dist = dist.Categorical(logits=act_logits)
            original_acts = torch.argmax(act_logits, dim=-1)
            acts = original_acts

        acts_list = acts.tolist()
        return acts_list

    def compute_sp_advantages(self, obs, n_obs, acts, dones, rews):
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
        baseline_values1 = []
        pred_values1 = []
        baseline_values2 = []
        pred_values2 = []

        # Initialize target diversity at end of episode to None
        pred_value1 = None
        pred_value2 = None
        for idx in reversed(range(obs_length)):
            obs_idx = obs[:, idx, :, :]
            n_obs_idx = n_obs[:, idx, :, :]
            sp_rl_rew1 = rews[:, idx, 0]
            sp_rl_rew2 = rews[:, idx, 1]
            sp_rl_done = dones[:, idx]

            if idx == obs_length - 1:
                # Get predicted returns from player 1 and 2's joint action value model
                pred_value1, pred_value2 = self.separate_critic_eval(n_obs_idx)

            pred_value1 = (
                    sp_rl_rew1.view(-1, 1) + (
                    self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * pred_value1)
            ).detach()

            pred_value2 = (
                    sp_rl_rew2.view(-1, 1) + (
                    self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * pred_value2)
            ).detach()

            baseline_value1, baseline_value2 = self.separate_critic_eval(obs_idx)

            pred_values1.append(pred_value1)
            baseline_values1.append(baseline_value1)
            pred_values2.append(pred_value2)
            baseline_values2.append(baseline_value2)
            
        # Combine lists into a single tensor for actor loss from SP data
        all_baselines1 = torch.cat(baseline_values1, dim=0).squeeze(-1)
        all_preds1 = torch.cat(pred_values1, dim=0).squeeze(-1)
        all_baselines2 = torch.cat(baseline_values2, dim=0).squeeze(-1)
        all_preds2 = torch.cat(pred_values2, dim=0).squeeze(-1)

        return all_preds1.detach() - all_baselines1.detach(), all_preds2.detach() - all_baselines2.detach(), None, None

    def compute_sp_old_probs(self, obs, acts):
        """
            A function that computes the previous policy's (before update) probability of selecting an action.  Required for MAPPO update.
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return old_log_likelihoods: Previous policies' action log probability
        """
        batch_size = obs.size()[0]
        obs_length = obs.size()[1]
        
        # Initialize empty lists that saves every important data for actor loss computation
        action_likelihood = []

        for idx in reversed(range(obs_length)):
            acts_idx = acts[:, idx, :, :]
            obs_idx = torch.cat(
                [obs[:, idx, :, :], torch.eye(self.num_agents).repeat(batch_size, 1, 1).to(self.device)], dim=-1)

            action_logits = self.separate_act_select_old(obs_idx)
            action_distribution = dist.OneHotCategorical(logits=action_logits)
            action_likelihood.append(action_distribution.log_prob(acts_idx))

        # Combine lists into a single tensor for actor loss from SP data
        old_log_likelihoods = torch.cat(action_likelihood, dim=0)
        return old_log_likelihoods

    def compute_sp_actor_loss(self, obs, acts, advantages1, advantages2, old_log_likelihoods):
        """
            A function that computes the policy's loss function based on self-play interaction
            :param obs: Agent observation
            :param acts: Agent action
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
        action_log_likelihoods = torch.cat(action_likelihood, dim=0)
        action_log_likelihoods1 = action_log_likelihoods[:,0]
        action_log_likelihoods2 = action_log_likelihoods[:,1]
        action_old_log_likelihoods1 = old_log_likelihoods[:,0]
        action_old_log_likelihoods2 = old_log_likelihoods[:,1]

        entropy_loss = -(action_entropies).sum(dim=-1).mean()


        ratio1 = torch.exp(action_log_likelihoods1 - action_old_log_likelihoods1.detach())
        ratio2 = torch.exp(action_log_likelihoods2 - action_old_log_likelihoods2.detach())

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
        pol_loss = -(pol_list1.mean() + pol_list2.mean())
        return pol_loss, entropy_loss

    def compute_sp_critic_loss(
            self, obs_batch, n_obs_batch,
            acts_batch, sp_rew_batch, sp_done_batch
    ):
        """
            A function that computes critic loss based on self-play interaction
            :param obs: Agent observation
            :param n_obs: Agent next observation
            :param acts: Agent action
            :param dones: Agent done flag
            :param rews: Agent reward flag
            :return critic loss
        """
        batch_size = obs_batch.size()[0]
        obs_length = obs_batch.size()[1]

        # Store values separately for each agent. This one is for Agent 1.
        predicted_values1 = []
        target_values1 = []
        target_value1 = None

        # Store values separately for each agent. This one is for Agent 2.
        predicted_values2 = []
        target_values2 = []
        target_value2 = None

        for idx in reversed(range(obs_length)):
            obs_idx = obs_batch[:,idx,:,:]
            n_obs_idx = n_obs_batch[:,idx,:,:]

            sp_v_values1, sp_v_values2 = self.separate_critic_eval(obs_idx)
            sp_rl_rew1 = sp_rew_batch[:, idx, 0]
            sp_rl_rew2 = sp_rew_batch[:, idx, 1]
            sp_rl_done = sp_done_batch[:, idx]

            if idx == obs_length-1:
                target_value1, target_value2 = self.separate_target_critic_eval(n_obs_idx)

            target_value1 = (
                    sp_rl_rew1.view(-1, 1) + (self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_value1)
            ).detach()

            target_value2 = (
                    sp_rl_rew2.view(-1, 1) + (self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_value2)
            ).detach()

            predicted_values1.append(sp_v_values1)
            target_values1.append(target_value1)
            predicted_values2.append(sp_v_values2)
            target_values2.append(target_value2)

        predicted_values1 = torch.cat(predicted_values1, dim=0)
        all_target_values1 = torch.cat(target_values1, dim=0)
        predicted_values2 = torch.cat(predicted_values2, dim=0)
        all_target_values2 = torch.cat(target_values2, dim=0)

        sp_critic_loss1 = (0.5 * ((predicted_values1 - all_target_values1) ** 2)).mean()
        sp_critic_loss2 = (0.5 * ((predicted_values2 - all_target_values2) ** 2)).mean()
        return sp_critic_loss1 + sp_critic_loss2

    def update(self, batches, xp_batches):
        """
            A method that updates the joint policy model following sampled self-play and cross-play experiences.
            Contains calls to the smaller functions that compute actor and critic losses.
            :param batches: A batch of obses and acts sampled from self-play experience replay.
            :param xp_batches: A batch of experience from cross-play experience replay.
        """
        self.total_updates += 1

        # Get obs and acts batch and prepare inputs to model.
        obs_batch, acts_batch = torch.tensor(batches[0]).to(self.device), torch.tensor(batches[1]).to(self.device)
        sp_n_obs_batch = torch.tensor(batches[2]).to(self.device)
        sp_done_batch = torch.tensor(batches[3]).double().to(self.device)
        rewards_batch = torch.tensor(batches[4]).double().to(self.device)

        sp_advantages1, sp_advantages2, _, _ = self.compute_sp_advantages(
            obs_batch, sp_n_obs_batch, acts_batch, sp_done_batch, rewards_batch
        )

        for pol, old_pol in zip(self.joint_policy, self.old_joint_policy):
            for target_param, param in zip(old_pol.parameters(), pol.parameters()):
                target_param.data.copy_(param.data)

        sp_old_log_probs = self.compute_sp_old_probs(
            obs_batch, acts_batch
        )

        for _ in range(self.config.train["epochs_per_update"]):
            self.actor_optimizer.zero_grad()

            # Compute SP Actor Loss
            sp_pol_loss, sp_action_entropies = self.compute_sp_actor_loss(
                obs_batch, acts_batch, sp_advantages1, sp_advantages2, sp_old_log_probs
            )
            total_actor_loss = sp_pol_loss + sp_action_entropies * self.config.loss_weights["entropy_regularizer_loss"]
            total_actor_loss.backward()

            if self.config.train['max_grad_norm'] > 0:
                for model in self.joint_policy:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.train['max_grad_norm'])

            self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        # Compute SP Critic Loss
        sp_critic_loss = self.compute_sp_critic_loss(
            obs_batch, sp_n_obs_batch, acts_batch, rewards_batch, sp_done_batch
        )

        total_critic_loss = sp_critic_loss * self.config.loss_weights["sp_val_loss_weight"]

        # Write losses to logs
        self.next_log_update += self.logger.train_log_period
        train_step = (self.total_updates-1) * self.logger.steps_per_update
        self.logger.log_item("Train/sp/actor_loss", sp_pol_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/sp/critic_loss", sp_critic_loss,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.log_item("Train/sp/entropy", sp_action_entropies,
                                 train_step=train_step, updates=self.total_updates-1)
        self.logger.commit()

        # Backpropagate critic loss
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

    def hard_copy(self):
        for idx in range(len(self.joint_action_value_functions)):
            for target_param, param in zip(self.target_joint_action_value_functions[idx].parameters(), self.joint_action_value_functions[idx].parameters()):
                target_param.data.copy_(param.data)

    def soft_copy(self, tau=0.001):
        for idx in range(len(self.joint_action_value_functions)):
            for target_param, param in zip(self.target_joint_action_value_functions[idx].parameters(), self.joint_action_value_functions[idx].parameters()):
                target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def save_model(self, int_id, save_model=False):
        """
            A method to save model parameters.
            :param int_id: Integer indicating ID of the checkpoint.
        """
        if not save_model:
            return

        for id, model in enumerate(self.joint_policy):
            torch.save(model.state_dict(), f"models/model_{id}_{int_id}.pt")

        for idx in range(len(self.joint_action_value_functions)):
            torch.save(self.joint_action_value_functions[idx].state_dict(),
                       f"models/model_{int_id}-action-value-{idx}.pt")

        for idx in range(len(self.target_joint_action_value_functions)):
            torch.save(self.target_joint_action_value_functions[idx].state_dict(),
                       f"models/model_{int_id}-target-action-value-{idx}.pt")

        torch.save(self.actor_optimizer.state_dict(),
                   f"models/model_{int_id}-act-optim.pt")

        torch.save(self.critic_optimizer.state_dict(),
                   f"models/model_{int_id}-crit-optim.pt")

    def load_model(self, int_id, overridden_model_dir=None):
        """
        A method to load stored models to be used by agents
        """

        if self.mode == "train":
            model_dir = self.config['load_dir']

            for id, model in enumerate(self.joint_policy):
                model.load_state_dict(
                    torch.load(f"{model_dir}/models/model_{id}_{int_id}.pt")
                )

            for id, model in enumerate(self.old_joint_policy):
                model.load_state_dict(
                    torch.load(f"{model_dir}/models/model_{id}_{int_id}.pt")
                )

            for id, model in enumerate(self.joint_action_value_functions):
                self.joint_action_value_functions[id].load_state_dict(
                    torch.load(f"{model_dir}/models/model_{int_id}-action-value-{id}.pt")
                )

            for id, model in enumerate(self.joint_action_value_functions):
                self.target_joint_action_value_functions.load_state_dict(
                    torch.load(f"{model_dir}/models/model_{int_id}-target-action-value-{id}.pt")
                )

            self.actor_optimizer.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-act-optim.pt")
            )

            self.critic_optimizer.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-crit-optim.pt")
            )

        else:
            model_dir = self.config.env['model_load_dir']
            if not overridden_model_dir is None:
                model_dir = overridden_model_dir

            for id, model in enumerate(self.joint_policy):
                model.load_state_dict(
                    torch.load(f"{model_dir}/model_{id}_{int_id}.pt", map_location=self.device)
                )