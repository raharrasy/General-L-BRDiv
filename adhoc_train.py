import gym
import random
from iterative_rps import rps_v0
from iterative_rps import n_agent_rps_v0
from lbforaging import lbf_v0
from lbforaging import lbf_simple_v0
from lbforaging import lbf_general_v0
from stag_hunt import state_based_n_agent_stag_hunt_v0
from stag_hunt import n_agent_stag_hunt_v0
import torch
import string
import json
from utils import PettingZooVectorizationParallelWrapper
import numpy as np
from ExpReplay import EpisodicExperienceReplay
from AdhocAgent import AdhocAgent
from NAgentLBRDivAgentPopulations import NAgentLBRDivAgentPopulations as Agents
from gymnasium.spaces import Discrete, Box
from utils import create_radar_plot
from scipy.special import softmax
# from train import Logger
import os
import wandb
from omegaconf import OmegaConf


class AdhocTraining(object):
    def __init__(self, config):
        """
            Constructor for a class that trains AHT agents based on generated teammates
                Args:
                    config : A dictionary containing required hyperparameters for AHT training
        """
        self.config = config
        self.device = torch.device("cuda" if config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
        self.env_name = config.env["name"]
        self.with_open_ended_learning = self.config.train.get("with_open_ended_learning", False)

        self.logger = Logger(config)
        #self.logger= None

        # Other experiment related variables
        self.exp_replay = None

        self.sp_selected_agent_idx = None

        self.stored_obs = None
        self.stored_nobs = None
        self.prev_cell_values = None
        self.prev_agent_actions = None
        self.prev_rewards = None
        self.agent_representation_list = None
        self.total_eps_per_thread = None

    def env_select(self, env_name):
        if env_name == "rps-v0":
            return rps_v0
        if env_name == "n-agent-rps-v0":
            return n_agent_rps_v0
        elif env_name == "lbf-v0":
            return lbf_v0
        elif env_name == "lbf-simple-v0":
            return lbf_simple_v0
        elif env_name == "lbf-general-v0":
            return lbf_general_v0
        elif env_name == "stag-hunt-simple-v0":
            return n_agent_stag_hunt_v0
        elif env_name == "stag-hunt-general-v0":
            return state_based_n_agent_stag_hunt_v0
        raise Exception('Currently unsupported env!')
    
    def get_min_max_performance(self, env_name):
        if env_name == "lbf-simple-v0":
            return [0] * self.config.populations["num_populations"], [3] * self.config.populations["num_populations"]
        raise Exception('Currently unsupported env!')

    def get_obs_sizes(self, obs_space):
        """
            Method to get the size of the envs' obs space and length of obs features. Must be defined for every envs.
        """
        input_shape_with_pop_idx = [self.config.populations["num_agents_per_population"]]
        input_shape_with_pop_idx.append(obs_space.n if type(obs_space) is Discrete else obs_space.shape[-1])
        input_shape_with_pop_idx[-1] += self.config.populations["num_populations"]
        num_features_with_pop_idx = input_shape_with_pop_idx[-1]
        return input_shape_with_pop_idx, num_features_with_pop_idx
    
    def get_obs_sizes_aht(self, obs_space, act_sizes):
        """
            Method to get the size of the envs' obs space and length of obs features. Must be defined for every envs.
        """
        input_shape_with_pop_idx = [self.config.populations["num_agents_per_population"]]
        input_shape_with_pop_idx.append(obs_space.n if type(obs_space) is Discrete else obs_space.shape[-1])
        input_shape_with_pop_idx[-1] += act_sizes
        num_features_with_pop_idx = input_shape_with_pop_idx[-1]
        return input_shape_with_pop_idx, num_features_with_pop_idx

    def create_directories(self):
        """
            A method that creates the necessary directories for storing resulting logs & parameters.
        """
        if not os.path.exists("adhoc_model"):
            os.makedirs("adhoc_model")

        if self.config.logger["store_video"]:
            os.makedirs("adhoc_videos")

    def to_one_hot_population_id(self, indices, total_populations=None):
        """
            A method that converts population ID to an one-hot-ID  format.
        """
        if total_populations == None:
            num_pops = self.config.populations["num_populations"]
        else:
            num_pops = total_populations

        pop_indices = np.asarray(indices).astype(int)
        one_hot_ids = np.eye(num_pops)[pop_indices]

        return one_hot_ids
    
    def postprocess_obs(self, obs):
        reshaped_obs = np.concatenate([np.expand_dims(ob, axis=1) for ob in obs.values()], axis=1)
        return reshaped_obs

    def postprocess_acts(self, acts, env):
        all_acts = {}
        acts = np.asarray(acts)
        for i, a_id in enumerate(env.possible_agents):
            all_acts[a_id] = acts[:, i].tolist()

        return all_acts
    
    def postprocess_others(self, obs, rewards, terminations, truncations, infos):
        reshaped_obs = np.concatenate([np.expand_dims(ob, axis=1) for ob in obs.values()], axis=1)
        rew = np.concatenate([np.expand_dims(rew, axis=-1) for rew in rewards.values()], axis=-1)
        term = np.any(np.concatenate([np.expand_dims(ter, axis=-1) for ter in terminations.values()], axis=-1), axis=-1)
        trunc = np.any(np.concatenate([np.expand_dims(tru, axis=-1) for tru in truncations.values()], axis=-1), axis=-1)

        return reshaped_obs, rew, term, trunc
    
    def select_sp_agents(self, num_envs, num_agent_populations, default_agent_populations=None):
        if default_agent_populations is None:
            if not self.with_open_ended_learning:
                return [np.random.choice(list(range(num_agent_populations)), 1)[0] for _ in range(num_envs)]
            else:
                return [np.random.choice(list(range(num_agent_populations)), 1, p=self.sampling_probs)[0] for _ in range(num_envs)]
            return [np.random.choice(list(range(num_agent_populations)), 1)[0] for _ in range(num_envs)]
        return [default_agent_populations for _ in range(num_envs)]

    def adhoc_data_gathering(self, env, adhoc_agent, agent_population, tuple_obs_size, act_sizes_all, total_generated_agents=None, default_agent_population=None):
        """
            A method that, given an environment, AHT agent, and \Pi^{\text{train}}, gathers interaction data between
            an AHT agent and its teammates.
        """

        target_timesteps_elapsed = self.config.train["timesteps_per_update"]
        num_envs = env.num_envs
        timesteps_elapsed = 0

        self.agent_representation_list = []

        num_trained_agents = self.config.populations["num_populations"]
        if not total_generated_agents is None:
            num_trained_agents = total_generated_agents
        

        if not self.sp_selected_agent_idx:
            # Sample population ids in case we don't know which populations are involved in SP
            self.sp_selected_agent_idx = self.select_sp_agents(
                num_envs, num_trained_agents, default_agent_populations=default_agent_population
            )

        real_obs_header_size = [num_envs, target_timesteps_elapsed]
        act_header_size = [num_envs, target_timesteps_elapsed]
        batch_size = num_envs

        real_obs_header_size.extend(list(tuple_obs_size))
        act_header_size.extend(list(act_sizes_all))

        stored_real_obs = np.zeros(real_obs_header_size)
        stored_next_real_obs = np.zeros(real_obs_header_size)
        stored_acts = np.zeros(act_header_size)
        stored_rewards = np.zeros([num_envs, target_timesteps_elapsed, self.config.populations["num_agents_per_population"]])
        stored_dones = np.zeros([num_envs, target_timesteps_elapsed])
        stored_truncs = np.zeros([num_envs, target_timesteps_elapsed])

        cell_values = self.prev_cell_values
        agent_prev_action = self.prev_agent_actions
        agent_prev_rews = self.prev_rewards

        if self.total_eps_per_thread is None:
            self.total_eps_per_thread = np.zeros([num_envs])

        while timesteps_elapsed < target_timesteps_elapsed:
            one_hot_id_shape = list(self.stored_obs.shape)[:-1]
            one_hot_ids = self.to_one_hot_population_id(
                np.expand_dims(np.asarray(self.sp_selected_agent_idx), axis=-1) * np.ones(one_hot_id_shape))

            # Decide agent's action based on model and execute.
            real_input = np.concatenate([self.stored_obs, one_hot_ids], axis=-1)
            acts = agent_population.decide_acts(real_input, True)

            # Compute teammate_representation
            if agent_prev_action is None:
                agent_prev_action = np.zeros([batch_size, act_sizes_all[0], act_sizes_all[-1]])

            if agent_prev_rews is None:
                agent_prev_rews = np.zeros([batch_size, act_sizes_all[0], 1])

            encoder_representation_input = np.concatenate(
                [
                    self.stored_obs, agent_prev_action, agent_prev_rews
                ], axis=-1
            )

            agent_representation, cell_values = adhoc_agent.get_teammate_representation(
                encoder_representation_input, cell_values
            )

            self.agent_representation_list.append(agent_representation.unsqueeze(1))

            rl_agent_representation = agent_representation
            ah_obs = torch.tensor(self.stored_obs).double().to(self.device)[:, -1, :]
            ah_policy_input = torch.cat([ah_obs, rl_agent_representation], dim=-1)
            ah_agents_acts = adhoc_agent.decide_acts(ah_policy_input, True)

            for a1, a2 in zip(acts, ah_agents_acts):
                a1[-1] = a2
            dict_acts = self.postprocess_acts(acts, env = env)
            self.stored_nobs, rews, dones, trunc, infos = env.step(dict_acts)
            self.stored_nobs, rews, dones, trunc = self.postprocess_others(self.stored_nobs, rews, dones, trunc, infos)
            
            next_rews = np.tile(
                np.expand_dims(np.expand_dims(rews[:, -1], -1), -1), 
                (1, self.config.populations["num_agents_per_population"], 1)
            )

            # Store data from most recent timestep into tracking variables
            one_hot_acts = agent_population.to_one_hot(acts)
            next_encoder_representation_input = np.concatenate(
                [
                    self.stored_nobs, one_hot_acts, next_rews
                ], axis=-1
            )

            stored_real_obs[:, timesteps_elapsed] = encoder_representation_input[:, :, :-1]
            stored_next_real_obs[:, timesteps_elapsed] = next_encoder_representation_input[:, :, :-1]
            stored_acts[:, timesteps_elapsed] = one_hot_acts
            stored_rewards[:, timesteps_elapsed] = rews
            stored_dones[:, timesteps_elapsed] = dones
            stored_truncs[:, timesteps_elapsed] = trunc

            agent_prev_action = one_hot_acts
            agent_prev_rews = next_rews
            self.stored_obs = self.stored_nobs

            timesteps_elapsed += 1

            # TODO Change agent id in finished envs.
            for idx, flag in enumerate(zip(dones, trunc)):
                # If an episode collected by one of the threads ends...
                if flag[0] or flag[1]:
                    self.total_eps_per_thread[idx] += 1
                    if self.total_eps_per_thread[idx] % self.config.env_eval["eps_per_interaction"] == 0:
                        self.sp_selected_agent_idx[idx] = \
                        np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0]
                        cell_values[0][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                            self.device)
                        cell_values[1][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                            self.device)
                        agent_prev_action[idx] = np.zeros(list(agent_prev_action[idx].shape))
                        agent_prev_rews[idx] = np.zeros(list(agent_prev_rews[idx].shape))

        self.prev_cell_values = cell_values
        self.prev_agent_actions = agent_prev_action
        self.prev_rewards = agent_prev_rews

        encoder_representation_input = np.concatenate(
            [
                self.stored_obs, agent_prev_action, agent_prev_rews
            ], axis=-1
        )

        agent_representation, _ = adhoc_agent.get_teammate_representation(
            encoder_representation_input, cell_values
        )

        self.agent_representation_list.append(agent_representation.unsqueeze(1))
        for r_obs, nr_obs, acts, rewards, dones, trun in zip(stored_real_obs, stored_next_real_obs, stored_acts,
                                                       stored_rewards, stored_dones, stored_truncs):
            self.exp_replay.add_episode(r_obs, acts, rewards, dones, trun, nr_obs)

    def eval_aht_policy_performance(
            self, adhoc_agent, agent_population, logger, logging_id, 
            eval=False, pop_size=None, num_agents_per_pop=None, make_video=False):
        """
            A method to evaluate the resulting performance of a trained agent population model when 
            dealing with its best-response policy.
            :param agent_population: An collection of agent policies whose SP returns are evaluated
            :param logger: A wandb logger used for writing results.
            :param logging_id: Checkpoint ID for logging.
        """
        
        env1 = self.env_select(self.config.env["name"]).parallel_env(render_mode=None)
        act_sizes = env1.action_space(env1.possible_agents[0]).n
        act_sizes_all = (self.config.populations["num_agents_per_population"], act_sizes) 

        if pop_size is None:
            num_pops = self.config.populations["num_populations"]
            num_agents_per_pop = self.config.populations["num_agents_per_population"]
        else:
            num_pops = pop_size
            num_agents_per_pop = num_agents_per_pop
            
        returns = np.zeros((num_pops, num_pops, self.config.populations["num_agents_per_population"]))

        all_performances = []
        for pop_id in range(num_pops):
            env_train = PettingZooVectorizationParallelWrapper(self.config.env["name"], n_envs=self.config.env.parallel["eval"])
            
            device = torch.device("cuda" if self.config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
            episodes_elapsed_per_thread = np.zeros([self.config.env.parallel["eval"]])
            total_returns_discounted = np.zeros([self.config.env.parallel["eval"], self.config.run["num_eval_episodes"]])
            total_returns_undiscounted = np.zeros([self.config.env.parallel["eval"], self.config.run["num_eval_episodes"]])
            # Initialize initial obs and states for model
            obs, _ = env_train.reset(seed=self.config.run["eval_seed"])
            obs = self.postprocess_obs(obs)
            
            time_elapsed = np.zeros([obs.shape[0], 1])

            # Set initial values for interaction
            cell_values = None
            agent_prev_action = np.zeros([self.config.env.parallel["eval"], act_sizes_all[0], act_sizes_all[-1]])
            agent_prev_rew = np.zeros([self.config.env.parallel["eval"], act_sizes_all[0], 1])
            
            while (any([k < self.config.run["num_eval_episodes"] for k in episodes_elapsed_per_thread])):
                #acts = agent_population.decide_acts(np.concatenate([obs, remaining_target, time_elapsed], axis=-1))
                one_hot_id_shape = list(obs.shape)[:-1]
                one_hot_ids = self.to_one_hot_population_id(pop_id*np.ones(one_hot_id_shape), total_populations=num_pops)

                # Decide agent's action based on model & target returns. Note that additional input concatenated to 
                # give population id info to policy.

                acts = agent_population.decide_acts(np.concatenate([obs, one_hot_ids], axis=-1), eval=eval)
                
                if agent_prev_action is None:
                    agent_prev_action = np.zeros([
                        self.config.env.parallel["eval"], 
                        act_sizes_all[0], act_sizes_all[-1]]
                    )

                if agent_prev_rew is None:
                    agent_prev_rew = np.zeros([
                        self.config.env.parallel["eval"], 
                        act_sizes_all[0], 1]
                    )

                encoder_representation_input = np.concatenate(
                    [
                        obs, agent_prev_action, agent_prev_rew
                    ], axis=-1
                )

                agent_representation, cell_values = adhoc_agent.get_teammate_representation(
                    encoder_representation_input, cell_values
                )

                rl_agent_representation = agent_representation.detach()
                ah_obs = torch.tensor(obs).double().to(self.device)[:, -1, :]
                ah_policy_input = torch.cat([ah_obs, rl_agent_representation], dim=-1)
                aht_agents_acts = adhoc_agent.decide_acts(ah_policy_input, True)
                for act, aht_act in zip(acts, aht_agents_acts):
                    act[-1] = aht_act
                acts = self.postprocess_acts(acts, env = env1)
                
                # Execute prescribed action
                n_obs, rews, dones, trunc, infos = env_train.step(acts)
                n_obs, rews, dones, trunc = self.postprocess_others(n_obs, rews, dones, trunc, infos)

                obs = n_obs
                time_elapsed = time_elapsed+1

                for idx, (flag0, flag1) in enumerate(zip(dones, trunc)):
                # If an episode collected by one of the threads ends...
                    if episodes_elapsed_per_thread[idx] < total_returns_undiscounted.shape[1]:
                        total_returns_undiscounted[idx][int(episodes_elapsed_per_thread[idx])] += rews[idx][-1]
                        total_returns_discounted[idx][int(episodes_elapsed_per_thread[idx])] += (
                            self.config.train["gamma"]**time_elapsed[idx][0]
                        )*rews[idx][-1]
                        time_elapsed[idx][0] += 1

                    if flag0 or flag1:
                        episodes_elapsed_per_thread[idx] += 1
                        if episodes_elapsed_per_thread[idx] + 1 < self.config.run["num_eval_episodes"] and episodes_elapsed_per_thread[idx] % self.config.env_eval["eps_per_interaction"] == 0:
                            
                            cell_values[0][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                                self.device)
                            cell_values[1][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                                self.device)
                            agent_prev_action[idx] = np.zeros(list(agent_prev_action[idx].shape))
                            agent_prev_rew[idx] = np.zeros(list(agent_prev_rew[idx].shape))
                        episodes_elapsed_per_thread[idx] = min(episodes_elapsed_per_thread[idx] + 1, self.config.run["num_eval_episodes"])
                        time_elapsed[idx][0] = 0

            logger.log_item(
                f"Returns/train/discounted_{pop_id}",
                np.mean(total_returns_discounted),
                checkpoint=logging_id)
            logger.log_item(
                f"Returns/train/nondiscounted_{pop_id}",
                np.mean(total_returns_undiscounted),
                checkpoint=logging_id)
            all_performances.append(np.mean(total_returns_undiscounted))
        
        data = {
            "type_names": ["T-{}".format(id) for id in range(num_pops)],
            "performances": [
                all_performances
            ]
        }

        env_min, env_max = self.get_min_max_performance(
            self.config.env["name"]
        )

        if self.with_open_ended_learning:
            sample_logits = [max(0, max_perf - perf) for max_perf, perf in zip(env_max, all_performances)]
            self.sampling_probs = softmax(np.asarray(sample_logits))
        create_radar_plot(
            data, 
            "radar_plot_{}.png".format(logging_id),
            ["Agent 1"], 
            ['b'],
            env_max, 
            lower_bound_performance=env_min,
            base_threshold=0.0,
        )
	

    
    def run(self):
        """
            A method that encompasses the main training loop for population-based training.
        """

        # Initialize environment, agent population model & experience replay based on obs vector sizes
        env1 = self.env_select(
            self.config.env["name"]
        ).parallel_env(render_mode=None)
        self.original_env = env1
        
        # Get act space data
        act_sizes = env1.action_space(env1.possible_agents[0]).n
        act_sizes_all = (self.config.populations["num_agents_per_population"], act_sizes)   

        # Get obs space data
        conf_obs_sizes, conf_num_obs_features = self.get_obs_sizes(env1.observation_space(env1.possible_agents[0]))
        
        obs_sizes, num_obs_features = self.get_obs_sizes_aht(env1.observation_space(env1.possible_agents[0]), act_sizes)
        tuple_obs_size = tuple(obs_sizes)

        env = PettingZooVectorizationParallelWrapper(self.config.env["name"], n_envs=self.config.env.parallel["adhoc_collection"])
        device = self.device

        # Create directories for logging
        self.create_directories()
        pop_class = Agents
        
        # Initialize implemented agent population
        agent_population = pop_class(
            conf_num_obs_features, conf_obs_sizes[0], self.config.populations["num_populations"], self.config, act_sizes, device, self.logger, mode="eval"
        )
        agent_population.load_model(self.config.run["model_id"])
        
        # Initialize experience replays that collect learning data
        self.exp_replay = EpisodicExperienceReplay(
            tuple_obs_size, act_sizes_all, 
            max_episodes=self.config.env.parallel["adhoc_collection"], 
            max_eps_length=self.config.train["timesteps_per_update"]
        )

        adhoc_agent = AdhocAgent(
            obs_sizes[-1],
            self.config, act_sizes, device, self.logger
        )

        # TODO Fix evaluation later
        if self.config.run["load_from_checkpoint"] == -1:
            adhoc_agent.save_model(0, save_model=self.logger.save_model)
            self.eval_aht_policy_performance(adhoc_agent, agent_population, self.logger, 0)
        else:
            adhoc_agent.load_model(self.config.run["load_from_checkpoint"])

        # Compute number of episodes required for training in each checkpoint.
        checkpoints_elapsed = self.config.run["load_from_checkpoint"] if self.config.run["load_from_checkpoint"] != -1 else 0
        total_checkpoints = self.config.run["total_checkpoints"]
        timesteps_per_checkpoint = self.config.run["num_timesteps"]//(total_checkpoints*(self.config.env.parallel["adhoc_collection"]))

        for ckpt_id in range(checkpoints_elapsed, total_checkpoints):
            # Record number of episodes that has elapsed in a checkpoint

            self.stored_obs, _= env.reset()
            self.stored_obs = self.postprocess_obs(self.stored_obs)

            timesteps_elapsed = 0
            while timesteps_elapsed < timesteps_per_checkpoint:
                # Do Policy update

                self.adhoc_data_gathering(
                    env, adhoc_agent, agent_population, tuple_obs_size, act_sizes_all
                )

                timesteps_elapsed += self.config.train["timesteps_per_update"]
                batches = self.exp_replay.sample_all()

                all_agent_representations = torch.cat(self.agent_representation_list, dim=1)
                adhoc_agent.update(batches, all_agent_representations)
                self.prev_cell_values = (
                    self.prev_cell_values[0].detach(),
                    self.prev_cell_values[1].detach()
                )

                self.exp_replay = EpisodicExperienceReplay(
                    tuple_obs_size, list(act_sizes_all),
                    max_episodes=self.config.env.parallel["adhoc_collection"],
                    max_eps_length=self.config.train["timesteps_per_update"]
                )

            # Eval policy after sufficient number of episodes were collected.
            adhoc_agent.save_model(ckpt_id + 1, save_model=self.logger.save_model)
            self.eval_aht_policy_performance(adhoc_agent, agent_population, self.logger, ckpt_id+1)
            if self.logger:
                self.logger.commit()

class Logger:    
    def __init__(self, config):    
        logger_period = config.logger.logger_period    
        self.steps_per_update = (config.env.parallel.adhoc_collection) * config.train.timesteps_per_update    
        self.save_model = config.logger.get("save_model", False)
        if logger_period < 1:    
            # Frequency    
            self.train_log_period = int(logger_period * config.run.num_timesteps // self.steps_per_update) + 1    
        else:    
            # Period    
            self.train_log_period = logger_period    
        self.verbose = config.logger.get("verbose", False)    
        self.run = wandb.init(    
            project=config.logger.project,    
            entity=config.logger.entity,    
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),    
            tags=config.logger.get("tags", None),    
            notes=config.logger.get("notes", None),    
            group=config.logger.get("group", None),    
            mode=config.logger.get("mode", None),    
            reinit=True,    
        )    
        self.define_metrics()    
    def log(self, data, step=None, commit=False):    
        wandb.log(data, step=step, commit=commit)    
    def log_item(self, tag, val, step=None, commit=False, **kwargs):    
        self.log({tag: val, **kwargs}, step=step, commit=commit)    
        if self.verbose:    
            print(f"{tag}: {val}")    
    def commit(self):    
        self.log({}, commit=True)    
    def define_metrics(self):    
        wandb.define_metric("train_step")    
        wandb.define_metric("checkpoint")    
        wandb.define_metric("Train/*", step_metric="train_step")    
        wandb.define_metric("Returns/*", step_metric="checkpoint")    



