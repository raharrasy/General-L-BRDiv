from MAPPOAgentPopulations import MAPPOAgentPopulations
import torch
from torch import optim, nn
import torch.distributions as dist

# TODO Add truncated bootstrapping
class TrajeDiAgentPopulations(MAPPOAgentPopulations):
    """
        A class that implements a MAPPO-based TrajeDi implementation.
        It simply adds an additional trajectory diversity loss on top of MAPPO. 
    """
    def __init__(self, obs_size, num_agents, num_populations, configs, act_sizes, device, logger, mode="train"):
        super().__init__(obs_size, num_agents, num_populations, configs, act_sizes, device, logger, mode)


    def compute_jsd_loss(self, obs_batch, acts_batch):
        """
            Legacy function from L-BRDiv to compute JSD
        """
        comparator_prob = None
        batch_size, num_steps, num_agents = obs_batch.size()[0], obs_batch.size()[1], obs_batch.size()[2]

        action_probs_per_population = []
        agent_real_ids = obs_batch[:, :, :, self.obs_size - self.num_populations: self.obs_size].argmax(dim=-1)

        # This code computes log(pi_i) for all possible populations
        for idx in range(self.num_populations):
            original_states = obs_batch[:, :, :, :self.obs_size - self.num_populations]
            original_states = original_states.view(batch_size*num_steps, num_agents, -1)

            pop_annot = torch.zeros_like(
                obs_batch.view(
                    batch_size*num_steps, num_agents, -1
                )[:, :, self.obs_size - self.num_populations:self.obs_size]).double().to(self.device)
            pop_annot[:, :, idx] = 1.0

            comparator_input = torch.cat([original_states, pop_annot, torch.eye(self.num_agents).repeat(original_states.size()[0],1,1)], dim=-1)
            comparator_act_logits = self.separate_act_select(comparator_input)
            comparator_act_logits = comparator_act_logits.view(batch_size, num_steps, num_agents, -1)

            action_logits = dist.OneHotCategorical(logits=comparator_act_logits).log_prob(acts_batch)
            action_probs_per_population.append(action_logits.unsqueeze(-1))

            if comparator_prob is None:
                comparator_prob = torch.exp(action_logits)
            else:
                comparator_prob = comparator_prob + torch.exp(action_logits)

        # This code evaluates log(pi_i) for the population generating the trajectory
        action_logits_per_population = torch.cat(action_probs_per_population, dim=-1)
        temp_pi= torch.gather(action_logits_per_population, -1, agent_real_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        log_pi_i = temp_pi.sum(dim=-1)

        # This code evaluates log(\hat{pi}) of the trajectory
        temp_pi_hat = action_logits_per_population.sum(dim=-2).sum(dim=-2)
        log_pi_hat = torch.log(torch.exp(temp_pi_hat).mean(dim=-1))

        summed_term_list = []
        separate_delta_list = []
        for t in range(num_steps):
            multiplier = self.config.train["gamma_act_jsd"]**(torch.abs(t-torch.tensor(list(range(num_steps))).to(self.device)))
            multiplier = multiplier.unsqueeze(0).repeat(batch_size,1)

            delta_hat_var = action_logits_per_population.sum(dim=-2)
            separate_deltas = (temp_pi * multiplier).sum(dim=-1)
            log_average_only_delta = (delta_hat_var * multiplier.unsqueeze(-1).repeat(1, 1, self.num_populations)).sum(dim=-2)
            average_only_delta = torch.log(torch.exp(log_average_only_delta).mean(dim=-1))

            separate_delta_list.append(separate_deltas.unsqueeze(-1))
            summed_term_list.append(average_only_delta.unsqueeze(-1))

        # This computes \delta_{t}
        stacked_summed_term_list = torch.cat(summed_term_list, dim=-1)
        # This computes \delta_{i,t}
        stacked_separate_delta_list = torch.cat(separate_delta_list, dim=-1)

        # This is first line in Eq 1
        pi_hat_per_pi_i = torch.exp(log_pi_hat - log_pi_i)
        term1 = pi_hat_per_pi_i.unsqueeze(-1).repeat(1, num_steps) * torch.exp(stacked_separate_delta_list)
        final_term1 = term1.detach() * stacked_separate_delta_list

        # This is second line in Eq 6
        term2_mult = torch.exp(stacked_summed_term_list) - (stacked_separate_delta_list/self.num_populations)
        final_term2 = term2_mult.detach() * log_pi_i.unsqueeze(-1).repeat(1, num_steps)

        # Averaged over time and batch
        jsd_loss = (final_term1 + final_term2).mean(dim=-1).mean()

        return jsd_loss
    
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
            act_diff_log_mean = self.compute_jsd_loss(obs_batch, acts_batch)
            total_actor_loss = sp_pol_loss + sp_action_entropies  * self.config.loss_weights["entropy_regularizer_loss"] + act_diff_log_mean*self.config.loss_weights["jsd_weight"]
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
        self.logger.log_item("Train/sp/jsd_loss", act_diff_log_mean,
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