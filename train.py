from time import time
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.spaces import Discrete, Box
import random
from iterative_rps import rps_v0
from iterative_rps import n_agent_rps_v0
from lbforaging import lbf_v0
from lbforaging import lbf_simple_v0
from utils import PettingZooVectorizationParallelWrapper
import torch
import string
import numpy as np
from ExpReplay import EpisodicExperienceReplay
from MAPPOAgentPopulations import MAPPOAgentPopulations
from NAgentLBRDivAgentPopulations import NAgentLBRDivAgentPopulations
import os
import wandb
from omegaconf import OmegaConf

class DiversityTraining(object):
    """
        A class that runs an experiment on learning with Upside Down Reinforcement Learning (UDRL).
    """
    def __init__(self, config):
        """
            Constructor for UDRLTraining class
                Args:
                    config : A dictionary containing required hyperparameters for UDRL training
        """
        self.config = config
        cuda_device = "cuda"
        if config.run['device_id'] != -1:
            cuda_device = cuda_device + ":" + str(config.run['device_id'])
        self.device = torch.device(cuda_device if config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
        self.env_name = config.env["name"]
        self.original_env = None

        self.logger = Logger(config)

        # Other experiment related variables
        self.exp_replay = None
        self.cross_play_exp_replay = None

        self.sp_selected_agent_idx = None
        self.xp_selected_agent_idx_p1 = None
        self.xp_selected_agent_idx_p2 = None

        self.stored_obs = None
        self.stored_nobs = None

        self.stored_obs_xp = None
        self.stored_nobs_xp = None

    def env_select(self, env_name):
        if env_name == "rps-v0":
            return rps_v0
        if env_name == "n-agent-rps-v0":
            return n_agent_rps_v0
        elif env_name == "lbf-v0":
            return lbf_v0
        elif env_name == "lbf-simple-v0":
            return lbf_simple_v0
        raise Exception('Currently unsupported env!')

    def algorithm_select(self, alg_name):
        if alg_name == "FCP":
            return MAPPOAgentPopulations
        elif alg_name == "L-BRDiv":
            return NAgentLBRDivAgentPopulations
        raise Exception('Currently unsupported algorithm!')

    def get_obs_sizes(self, obs_space):
        """
            Method to get the size of the envs' obs space and length of obs features.
            Note that we append additional one-hot index to the end of original observation.
            This additional index is done to help distinguish inputs for different agents.
        """

        input_shape_with_pop_idx = [self.config.populations["num_agents_per_population"]]
        input_shape_with_pop_idx.append(obs_space.n if type(obs_space) is Discrete else obs_space.shape[-1])
        input_shape_with_pop_idx[-1] += self.config.populations["num_populations"]
        num_features_with_pop_idx = input_shape_with_pop_idx[-1]
        return input_shape_with_pop_idx, num_features_with_pop_idx

    def create_directories(self):
        """
            A method that creates the necessary directories for storing resulting logs & parameters.
        """
        if not os.path.exists("models"):
            os.makedirs("models")

        if self.config.logger["store_video"]:
            os.makedirs("videos")

    def to_one_hot_population_id(self, indices, total_populations=None):
        """
            Method to turn population id to one-hot representation.
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

    def eval_sp_policy_performance(self, agent_population, logger, logging_id, eval=False, pop_size=None, num_agents_per_pop=None, make_video=False):
        """
            A method to evaluate the resulting performance of a trained agent population model when 
            dealing with its best-response policy.
            :param agent_population: An collection of agent policies whose SP returns are evaluated
            :param logger: A wandb logger used for writing results.
            :param logging_id: Checkpoint ID for logging.
        """

        # Create env for policy eval

        # TODO Uncomment once fixes in PZoo is implemented
        # def make_video_env(env_name, video_path):
        #     env = gym.make(env_name)
        #     vc = VideoRecorder(env, path=video_path, enabled=True) 
        #     return env, vc
        
        env1 = self.env_select(self.config.env["name"]).parallel_env(render_mode=None)

        if pop_size is None:
            num_pops = self.config.populations["num_populations"]
            num_agents_per_pop = self.config.populations["num_agents_per_population"]
        else:
            num_pops = pop_size
            num_agents_per_pop = num_agents_per_pop
            
        returns = np.zeros((num_pops, num_pops, self.config.populations["num_agents_per_population"]))

        for pop_id in range(num_pops):
            env_train = PettingZooVectorizationParallelWrapper(self.config.env["name"], n_envs=self.config.env.parallel["eval"])
            
            # TODO Uncomment once fixes in PZoo is implemented
            # if make_video:
            #     video_env_list = [make_video_env(self.config.env["name"], os.getcwd()+f"/videos/{logging_id}_{pop_id}_{a_id}.mp4") for a_id in range(num_agents_per_pop)]
            #     video_paths = [os.getcwd()+f"/videos/{logging_id}_{pop_id}_{a_id}.mp4" for a_id in range(num_agents_per_pop)]

            # Initialize objects to track returns.
            device = torch.device("cuda" if self.config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
            num_dones = [0] * self.config.env.parallel["eval"]
            # Log per thread and per agent discounted returns
            per_worker_and_agent_rew = [[0.0 for _ in range(self.config.populations["num_agents_per_population"])] for _ in range(self.config.env.parallel["eval"])]
            # Log per thread and per agent undiscounted returns
            per_worker_and_agent_non_disc_rew = [[0.0 for _ in range(self.config.populations["num_agents_per_population"])] for _ in range(self.config.env.parallel["eval"])]

            # Initialize initial obs and states for model
            obs, _ = env_train.reset(seed=self.config.run["eval_seed"])
            obs = self.postprocess_obs(obs)
            
            time_elapsed = np.zeros([obs.shape[0], obs.shape[1], 1])
            avgs_all_agents = [[] for _ in range(self.config.populations["num_agents_per_population"])]
            avgs_non_disc_all_agents = [[] for _ in range(self.config.populations["num_agents_per_population"])]

            while (any([k < self.config.run["num_eval_episodes"] for k in num_dones])):
                #acts = agent_population.decide_acts(np.concatenate([obs, remaining_target, time_elapsed], axis=-1))
                one_hot_id_shape = list(obs.shape)[:-1]
                one_hot_ids = self.to_one_hot_population_id(pop_id*np.ones(one_hot_id_shape), total_populations=num_pops)

                # Decide agent's action based on model & target returns. Note that additional input concatenated to 
                # give population id info to policy.

                acts = agent_population.decide_acts(np.concatenate([obs, one_hot_ids], axis=-1), eval=eval)
                acts = self.postprocess_acts(acts, env = env1)
                
                # Execute prescribed action
                n_obs, rews, dones, trunc, infos = env_train.step(acts)
                n_obs, rews, dones, trunc = self.postprocess_others(n_obs, rews, dones, trunc, infos)

                obs = n_obs
                time_elapsed = time_elapsed+1

                # Log per thread returns
                for thread_id in range(self.config.env.parallel["eval"]):
                    for agent_id in range(self.config.populations["num_agents_per_population"]):
                        per_worker_and_agent_rew[thread_id][agent_id] = per_worker_and_agent_rew[thread_id][agent_id] + (self.config.train["gamma"]**(time_elapsed[thread_id][0][0]-1))*rews[thread_id][agent_id] 
                
                # Log per thread discounted returns
                for thread_id in range(self.config.env.parallel["eval"]):
                    for agent_id in range(self.config.populations["num_agents_per_population"]):
                        per_worker_and_agent_non_disc_rew[thread_id][agent_id] = per_worker_and_agent_non_disc_rew[thread_id][agent_id] + rews[thread_id][agent_id] 

                for idx, flag in enumerate(dones):
                    # If an episode in one of the threads ends...
                    if flag or trunc[idx]:
                        # Reset all relevant variables used in tracking and send logged returns in the finished thread to a storage
                        time_elapsed[idx] = np.zeros([obs.shape[1], 1])
                        if num_dones[idx] < self.config.run['num_eval_episodes']:
                            num_dones[idx] += 1
                            for a_id in range(self.config.populations["num_agents_per_population"]):
                                avgs_all_agents[a_id].append(per_worker_and_agent_rew[idx][a_id])
                                avgs_non_disc_all_agents[a_id].append(per_worker_and_agent_non_disc_rew[idx][a_id])
                            
                        for a_id in range(self.config.populations["num_agents_per_population"]):
                            per_worker_and_agent_rew[idx][a_id] = 0
                            per_worker_and_agent_non_disc_rew[idx][a_id] = 0

                      
            # Log achieved returns.
            for a_id in range(self.config.populations["num_agents_per_population"]):
                returns[pop_id, pop_id, a_id] = np.mean(avgs_all_agents[a_id])
            env_train.close()

            for a_id in range(self.config.populations["num_agents_per_population"]):
                logger.log_item(
                    f"Returns/sp/discounted_{pop_id}_{a_id}",
                    np.mean(avgs_all_agents[a_id]),
                    checkpoint=logging_id)
                logger.log_item(
                    f"Returns/sp/nondiscounted_{pop_id}_{a_id}",
                    np.mean(avgs_non_disc_all_agents[a_id]),
                    checkpoint=logging_id)

            # TODO Uncomment once fixes in PZoo is implemented  
            # if make_video:
            #     outs = [env.reset(seed=self.config.run["eval_seed"]) for env, vc in video_env_list]
            #     obs = np.asarray([o[0] for o in outs])
            #     for env, vc in video_env_list:
            #         vc.capture_frame()

            #     all_done = [False] * num_agents_per_pop
            #     while not all(all_done):
            #         one_hot_id_shape = list(obs.shape)[:-1]
            #         one_hot_ids = self.to_one_hot_population_id(pop_id*np.ones(one_hot_id_shape), total_populations=num_pops)

            #         agent_ids = np.tile(
            #             np.expand_dims(np.eye(num_agents_per_pop), axis=1), (1, one_hot_ids.shape[1], 1)
            #         )

            #         # Decide agent's action based on model & target returns. Note that additional input concatenated to 
            #         # give population id info to policy.
            #         real_input = np.concatenate([obs, one_hot_ids], axis=-1)
            #         acts = agent_population.decide_acts(real_input, eval=eval)

            #         outs = []
            #         vid_idx = 0
            #         for (env, vc), act in zip(video_env_list, acts):
            #             outs.append(env.step(act))
            #             if not all_done[vid_idx]:
            #                 vc.capture_frame()
            #             vid_idx += 1
            #         all_done = [done or a[2] for done, a in zip(all_done, outs)]
            #         obs = np.asarray([o[0] for o in outs])
                
            #     for idx in range(len(video_env_list)):
            #         video_env_list[idx][1].close()

            #     for idx in range(len(video_env_list)):
            #         print("Testing : ",video_paths[idx])
            #         wandb.log({f'video/vid_{pop_id}_{idx}': wandb.Video(video_paths[idx], fps=1, format="mp4")}, commit=True)
        
        return returns

    def eval_xp_policy_performance(self, agent_population, logger, logging_id, eval=False, pop_size=None):
        """
            A method to evaluate the resulting performance of a trained agent population model when 
            dealing with the best-response policy for other populations.
            :param agent_population: An collection of agent policies whose XP matrix is evaluated
            :param logger: A wandb logger used for writing results.
            :param logging_id: Checkpoint ID for logging.
        """

        env1 = self.env_select(self.config.env["name"]).parallel_env(render_mode=None)
        if pop_size is None:
            num_pops = self.config.populations["num_populations"]
        else:
            num_pops = pop_size
        
        returns = np.zeros((num_pops, num_pops,self.config.populations["num_agents_per_population"]))
        for pop_id in range(num_pops):
            for oppo_id in range(num_pops):
                if pop_id != oppo_id:
                    env_train = PettingZooVectorizationParallelWrapper(self.config.env["name"], n_envs=self.config.env.parallel["eval"])

                    # Initialize objects to track returns.
                    device = torch.device("cuda" if self.config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
                    num_dones = [0] * self.config.env.parallel["eval"]
                    # Log per thread and per agent discounted returns
                    per_worker_and_agent_rew = [[0.0 for _ in range(self.config.populations["num_agents_per_population"])] for _ in range(self.config.env.parallel["eval"])]
                    # Log per thread and per agent undiscounted returns
                    per_worker_and_agent_non_disc_rew = [[0.0 for _ in range(self.config.populations["num_agents_per_population"])] for _ in range(self.config.env.parallel["eval"])]

                    # Initialize initial obs and states for model
                    obs, _ = env_train.reset(seed=self.config.run["eval_seed"])
                    obs = self.postprocess_obs(obs)
                    time_elapsed = np.zeros([obs.shape[0], obs.shape[1], 1])
                    avgs_all_agents = [[] for _ in range(self.config.populations["num_agents_per_population"])]
                    avgs_non_disc_all_agents = [[] for _ in range(self.config.populations["num_agents_per_population"])]

                    while (any([k < self.config.run["num_eval_episodes"] for k in num_dones])):

                        one_hot_id_shape_indiv1 = list(obs.shape)[:-2]
                        one_hot_id_shape_indiv1.append(self.config.populations["num_agents_per_population"]-1)

                        one_hot_id_shape_indiv2 = list(obs.shape)[:-2]
                        one_hot_id_shape_indiv2.append(1)
                        # In case of XP, added IDs are different
                        one_hot_ids = self.to_one_hot_population_id(pop_id*np.ones(one_hot_id_shape_indiv1), total_populations=num_pops)
                        one_hot_ids2 = self.to_one_hot_population_id(oppo_id*np.ones(one_hot_id_shape_indiv2), total_populations=num_pops)
                        # Decide agent's action based on model & target returns. Note that additional input concatenated to 
                        # give population id info to policy.

                        selected_acts = agent_population.decide_acts(
                            np.concatenate([obs, np.concatenate([one_hot_ids, one_hot_ids2], axis=-2)], axis=-1), eval=eval
                        )
                        selected_acts = self.postprocess_acts(selected_acts, env = env1)
                        n_obs, rews, dones, trunc, infos = env_train.step(selected_acts)
                        n_obs, rews, dones, trunc = self.postprocess_others(n_obs, rews, dones, trunc, infos)

                        obs = n_obs
                        time_elapsed = time_elapsed + 1

                        # Log per thread returns
                        for thread_id in range(self.config.env.parallel["eval"]):
                            for agent_id in range(self.config.populations["num_agents_per_population"]):
                                per_worker_and_agent_rew[thread_id][agent_id] = per_worker_and_agent_rew[thread_id][agent_id] + (self.config.train["gamma"]**(time_elapsed[thread_id][0][0]-1))*rews[thread_id][agent_id] 
                        
                        # Log per thread undiscounted returns
                        for thread_id in range(self.config.env.parallel["eval"]):
                            for agent_id in range(self.config.populations["num_agents_per_population"]):
                                per_worker_and_agent_non_disc_rew[thread_id][agent_id] = per_worker_and_agent_non_disc_rew[thread_id][agent_id] + rews[thread_id][agent_id] 

                        for idx, flag in enumerate(dones):
                            # If an episode in one of the threads ends...
                            if flag or trunc[idx]:
                                time_elapsed[idx] = np.zeros([obs.shape[1], 1])
                                if num_dones[idx] < self.config.run['num_eval_episodes']:
                                    num_dones[idx] += 1
                                    for a_id in range(self.config.populations["num_agents_per_population"]):
                                        avgs_all_agents[a_id].append(per_worker_and_agent_rew[idx][a_id])
                                        avgs_non_disc_all_agents[a_id].append(per_worker_and_agent_non_disc_rew[idx][a_id])
                                    
                                for a_id in range(self.config.populations["num_agents_per_population"]):
                                    per_worker_and_agent_rew[idx][a_id] = 0
                                    per_worker_and_agent_non_disc_rew[idx][a_id] = 0

                    # Log achieved returns.
                    for a_id in range(self.config.populations["num_agents_per_population"]):
                        returns[pop_id, oppo_id, a_id] = np.mean(avgs_all_agents[a_id])
                    env_train.close()
                    
                    for a_id in range(self.config.populations["num_agents_per_population"]):
                        logger.log_item(
                            f"Returns/xp/discounted_{pop_id}_{oppo_id}_{a_id}",
                            np.mean(avgs_all_agents[a_id]),
                            checkpoint=logging_id)
                        logger.log_item(
                            f"Returns/xp/nondiscounted_{pop_id}_{oppo_id}_{a_id}",
                            np.mean(avgs_non_disc_all_agents[a_id]),
                            checkpoint=logging_id)
        return returns
    
    def select_sp_agents(self, num_envs, num_agent_populations, default_agent_populations=None):
        if default_agent_populations is None:
            return [np.random.choice(list(range(num_agent_populations)), 1)[0] for _ in range(num_envs)]
        return [default_agent_populations for _ in range(num_envs)]

    def select_xp_agents(self, num_agent_populations, default_agent_populations=None):

        if default_agent_populations is None:
            xp_selected_agent_idx_p1 = [np.random.choice(list(range(num_agent_populations)), 1)[0]
                                             for _ in range(self.config.env.parallel["xp_collection"])]
        else:
            xp_selected_agent_idx_p1 = [default_agent_populations for _ in range(self.config.env.parallel["xp_collection"])]

        xp_selected_agent_idx_p2 = []
        for idx in range(len(xp_selected_agent_idx_p1)):
            # Since this is for XP, agent ID 1 and 2 must be different
            sampled_pair = np.random.choice(list(range(num_agent_populations)), 1)[0]
            while sampled_pair == xp_selected_agent_idx_p1[idx]:
                sampled_pair = np.random.choice(list(range(num_agent_populations)), 1)[0]

            xp_selected_agent_idx_p2.append(sampled_pair)

        return xp_selected_agent_idx_p1, xp_selected_agent_idx_p2

    def self_play_data_gathering(self, env, agent_population, tuple_obs_size, act_sizes_all, total_num_agents=None, default_agent_population=None, epsilon=None):
        """
            Method to get self-play data for the agent_population.
            Data collection will commence for "timesteps_per_update" timesteps 
            (specified in config.train)
        """

        target_timesteps_elapsed = self.config.train["timesteps_per_update"]
        timesteps_elapsed = 0
        num_envs = env.num_envs

        num_trained_agents = self.config.populations["num_populations"]
        if not total_num_agents is None:
            num_trained_agents = total_num_agents

        if not self.sp_selected_agent_idx:
            # Sample population ids in case we don't know which populations are involved in SP
            self.sp_selected_agent_idx = self.select_sp_agents(num_envs, num_trained_agents, default_agent_populations=default_agent_population)

        real_obs_header_size = [num_envs, target_timesteps_elapsed]
        act_header_size = [num_envs, target_timesteps_elapsed]

        real_obs_header_size.extend(list(tuple_obs_size))
        act_header_size.extend(list(act_sizes_all))

        stored_real_obs = np.zeros(real_obs_header_size)
        stored_next_real_obs = np.zeros(real_obs_header_size)
        stored_acts = np.zeros(act_header_size)
        stored_rewards = np.zeros([num_envs, target_timesteps_elapsed, self.config.populations["num_agents_per_population"]])
        stored_dones = np.zeros([num_envs, target_timesteps_elapsed])
        stored_truncs = np.zeros([num_envs, target_timesteps_elapsed])

        while timesteps_elapsed < target_timesteps_elapsed:
            
            one_hot_id_shape = list(self.stored_obs.shape)[:-1]
            one_hot_ids = self.to_one_hot_population_id(
                np.expand_dims(np.asarray(self.sp_selected_agent_idx), axis=-1) * np.ones(one_hot_id_shape), 
                total_populations = num_trained_agents    
            )
            
            # Decide agent's action based on model and execute.
            real_input = np.concatenate([self.stored_obs, one_hot_ids], axis=-1)
            original_acts = agent_population.decide_acts(real_input, True, epsilon=epsilon)
            acts = self.postprocess_acts(original_acts, self.original_env)
            self.stored_nobs, rews, dones, trunc, infos = env.step(acts)
            self.stored_nobs, rews, dones, trunc = self.postprocess_others(self.stored_nobs, rews, dones, trunc, infos)
            real_n_input = np.concatenate([self.stored_nobs, one_hot_ids], axis=-1)

            # Store data from most recent timestep into tracking variables
            one_hot_acts = agent_population.to_one_hot(original_acts)

            stored_real_obs[:, timesteps_elapsed] = real_input
            stored_next_real_obs[:, timesteps_elapsed] = real_n_input
            stored_acts[:, timesteps_elapsed] = one_hot_acts
            stored_rewards[:, timesteps_elapsed, :] = rews
            stored_dones[:, timesteps_elapsed] = dones
            stored_truncs[:, timesteps_elapsed] = trunc

            # Store last observation in case data collection ends and we
            # want to resume collection in the next data collection step
            self.stored_obs = self.stored_nobs
            timesteps_elapsed += 1

            # TODO Change agent id in finished envs.
            for idx, flag in enumerate(dones):
                # If an episode collected by one of the threads ends...
                if flag or trunc[idx]:
                    # Resample the population ids involved in self-play data collection
                    self.sp_selected_agent_idx[idx] = np.random.choice(list(range(num_trained_agents)), 1)[0]
                    if not default_agent_population is None:
                        self.sp_selected_agent_idx[idx] = default_agent_population
                        
        for r_obs, nr_obs, acts, rewards, dones, trunc in zip(stored_real_obs, stored_next_real_obs, stored_acts, stored_rewards, stored_dones, stored_truncs):
            self.exp_replay.add_episode(r_obs, acts, rewards, dones, trunc, nr_obs)


    def cross_play_data_gathering(self, env, agent_population, tuple_obs_size, act_sizes_all, total_num_agents=None, default_agent_population=None, epsilon=None, iterative_eval_flag=False):
        """
            Method to get cross-play data for the agent_population.
            Data collection will commence for "timesteps_per_update" timesteps 
            (specified in config.train)
        """

        # Get required data from selected agents
        target_timesteps_elapsed = self.config.train["timesteps_per_update"]

        num_trained_agents = self.config.populations["num_populations"]
        if not total_num_agents is None:
            num_trained_agents = total_num_agents

        if (self.xp_selected_agent_idx_p1 is None) or (self.xp_selected_agent_idx_p2 is None):
            self.xp_selected_agent_idx_p1, self.xp_selected_agent_idx_p2 = self.select_xp_agents(num_trained_agents, default_agent_populations=default_agent_population)

        timesteps_elapsed = 0

        real_obs_header_size = [self.config.env.parallel["xp_collection"], target_timesteps_elapsed]
        act_header_size = [self.config.env.parallel["xp_collection"], target_timesteps_elapsed]

        real_obs_header_size.extend(tuple_obs_size)
        act_header_size.extend(list(act_sizes_all))

        stored_real_obs = np.zeros(real_obs_header_size)
        stored_next_real_obs = np.zeros(real_obs_header_size)
        stored_acts = np.zeros(act_header_size)
        stored_rewards = np.zeros([self.config.env.parallel["xp_collection"], target_timesteps_elapsed, self.config.populations["num_agents_per_population"]])
        stored_dones = np.zeros([self.config.env.parallel["xp_collection"], target_timesteps_elapsed])
        stored_trunc = np.zeros([self.config.env.parallel["xp_collection"], target_timesteps_elapsed])

        while timesteps_elapsed < target_timesteps_elapsed:
            one_hot_id_shape_indiv1 = list(self.stored_obs_xp.shape)[:-2]
            one_hot_id_shape_indiv1.append(self.config.populations["num_agents_per_population"]-1)

            one_hot_id_shape_indiv2 = list(self.stored_obs_xp.shape)[:-2]
            one_hot_id_shape_indiv2.append(1)
            one_hot_ids = self.to_one_hot_population_id(
                np.expand_dims(np.asarray(self.xp_selected_agent_idx_p1), axis=-1)*np.ones(one_hot_id_shape_indiv1),
                total_populations = num_trained_agents    
            )
            one_hot_ids2 = self.to_one_hot_population_id(
                np.expand_dims(np.asarray(self.xp_selected_agent_idx_p2), axis=-1)*np.ones(one_hot_id_shape_indiv2),
                total_populations = num_trained_agents
            )

            # decide actions
            real_input = np.concatenate([self.stored_obs_xp, np.concatenate([one_hot_ids, one_hot_ids2], axis=-2)], axis=-1)
            original_acts = agent_population.decide_acts(
                real_input, epsilon=epsilon, eval=iterative_eval_flag
            )
            acts = self.postprocess_acts(original_acts, self.original_env)
            self.stored_nobs_xp, rews, dones, trunc, infos = env.step(acts)
            self.stored_nobs_xp, rews, dones, trunc = self.postprocess_others(self.stored_nobs_xp, rews, dones, trunc, infos)

            next_one_hot_id_shape_indiv1 = list(self.stored_nobs_xp.shape)[:-2]
            next_one_hot_id_shape_indiv1.append(self.config.populations["num_agents_per_population"]-1)
            next_one_hot_id_shape_indiv2 = list(self.stored_nobs_xp.shape)[:-2]
            next_one_hot_id_shape_indiv2.append(1)

            next_one_hot_ids = self.to_one_hot_population_id(np.expand_dims(np.asarray(self.xp_selected_agent_idx_p1), axis=-1)*np.ones(next_one_hot_id_shape_indiv1), total_populations=num_trained_agents)
            next_one_hot_ids2 = self.to_one_hot_population_id(np.expand_dims(np.asarray(self.xp_selected_agent_idx_p2), axis=-1)*np.ones(next_one_hot_id_shape_indiv2), total_populations=num_trained_agents)
            
            next_real_input = np.concatenate([self.stored_nobs_xp, np.concatenate([next_one_hot_ids, next_one_hot_ids2], axis=-2)], axis=-1)

            # Store data from most recent timestep into tracking variables
            one_hot_acts = agent_population.to_one_hot(original_acts)
            self.stored_obs_xp = self.stored_nobs_xp

            stored_real_obs[:, timesteps_elapsed] = real_input
            stored_next_real_obs[:, timesteps_elapsed] = next_real_input
            stored_acts[:, timesteps_elapsed] = one_hot_acts
            stored_rewards[:, timesteps_elapsed, :] = rews
            stored_dones[:, timesteps_elapsed] = dones
            stored_trunc[:, timesteps_elapsed] = trunc

            timesteps_elapsed += 1

            for idx, flag in enumerate(dones):
                # If an episode collected by one of the threads ends...
                if flag or trunc[idx]:
                    # Resample populations ids involved in XP.
                    self.xp_selected_agent_idx_p1[idx] = np.random.choice(list(range(num_trained_agents)), 1)[0]
                    if not default_agent_population is None:
                        self.xp_selected_agent_idx_p1[idx] = default_agent_population
                    sampled_pairing = np.random.choice(list(range(num_trained_agents)), 1)[0]
                    while sampled_pairing == self.xp_selected_agent_idx_p1[idx]:
                        # Make sure ID1 and ID2 always differs.
                        sampled_pairing = np.random.choice(list(range(num_trained_agents)), 1)[0]
                    self.xp_selected_agent_idx_p2[idx] = sampled_pairing

        for cur_obs, acts, rew, done, trunc, next_obs in zip(stored_real_obs, stored_acts, stored_rewards, stored_dones, stored_trunc,
                                                      stored_next_real_obs):
            self.cross_play_exp_replay.add_episode(cur_obs, acts, rew, done, trunc, next_obs)

    def run(self):
        """
            A method that encompasses the main training loop for population-based training.
        """

        # Create logging directories & utilities
        def randomString(stringLength=10):
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for i in range(stringLength))

        # Initialize environment, agent population model & experience replay based on obs vector sizes
        env1 = self.env_select(
            self.config.env["name"]
        ).parallel_env(render_mode=None)
        self.original_env = env1

        # Get obs space data
        obs_sizes, num_obs_features = self.get_obs_sizes(env1.observation_space(env1.possible_agents[0]))

        # Get act space data
        act_sizes = env1.action_space(env1.possible_agents[0]).n
        act_sizes_all = (self.config.populations["num_agents_per_population"], act_sizes)        
        tuple_obs_size = tuple(obs_sizes)

        env = PettingZooVectorizationParallelWrapper(self.config.env["name"], n_envs=self.config.env.parallel["sp_collection"])
        if self.config.env.parallel["xp_collection"] != 0:
            env_xp = PettingZooVectorizationParallelWrapper(self.config.env["name"], n_envs=self.config.env.parallel["xp_collection"])

        device = self.device

        # Create directories for logging
        self.create_directories()
        pop_class = self.algorithm_select(self.config.train["method"] )
        
        # Initialize implemented agent population
        agent_population = pop_class(
            num_obs_features, obs_sizes[0], self.config.populations["num_populations"], self.config, act_sizes, device, self.logger
        )
        
        # Initialize experience replays that collect agent self-play and cross-play data
        self.exp_replay = EpisodicExperienceReplay(
            tuple_obs_size, act_sizes_all, max_episodes=self.config.env.parallel["sp_collection"], max_eps_length=self.config.train["timesteps_per_update"]
        )
        self.cross_play_exp_replay = EpisodicExperienceReplay(
            tuple_obs_size, act_sizes_all, max_episodes=self.config.env.parallel["xp_collection"], max_eps_length=self.config.train["timesteps_per_update"]
        )

        # Save randomly initialized NN or load from pre-existing parameters if specified in argparse.
        if self.config.run["load_from_checkpoint"] == -1:
            agent_population.save_model(0, save_model=self.logger.save_model)
            sp_mat = self.eval_sp_policy_performance(agent_population, self.logger, 0)
            xp_mat = self.eval_xp_policy_performance(agent_population, self.logger, 0)
            
            for a_id in range(self.config.populations["num_agents_per_population"]):
                self.logger.log_xp_matrix(f"Returns/xp_matrix{a_id}", sp_mat[:, :, a_id] + xp_mat[:, :, a_id], checkpoint=0)
        else:
            agent_population.load_model(self.config.run["load_from_checkpoint"])

        # Compute number of episodes required for training in each checkpoint.
        checkpoints_elapsed = self.config.run["load_from_checkpoint"] if self.config.run["load_from_checkpoint"] != -1 else 0
        total_checkpoints = self.config.run["total_checkpoints"]
        timesteps_per_checkpoint = self.config.run["num_timesteps"]//(total_checkpoints*(self.config.env.parallel["sp_collection"]+self.config.env.parallel["xp_collection"]))

        for ckpt_id in range(checkpoints_elapsed, total_checkpoints):
            # Record number of episodes that has elapsed in a checkpoint

            self.stored_obs, _= env.reset()
            self.stored_obs = self.postprocess_obs(self.stored_obs)
            if self.config.env.parallel["xp_collection"] !=0:
                self.stored_obs_xp, _ = env_xp.reset()
                self.stored_obs_xp = self.postprocess_obs(self.stored_obs_xp)


            timesteps_elapsed = 0
            while timesteps_elapsed < timesteps_per_checkpoint:
                # Do Policy update
                self.self_play_data_gathering(
                    env, agent_population, tuple_obs_size, act_sizes_all
                )

                if self.config.env.parallel["xp_collection"] != 0:
                    self.cross_play_data_gathering(
                        env_xp, agent_population, tuple_obs_size, act_sizes_all
                    )

                timesteps_elapsed += self.config.train["timesteps_per_update"]
                batches = self.exp_replay.sample_all()
                if self.config.env.parallel["xp_collection"] != 0:
                    batches_xp = self.cross_play_exp_replay.sample_all()
                else:
                    batches_xp = None

                agent_population.update(batches, batches_xp)
                self.exp_replay = EpisodicExperienceReplay(
                    tuple_obs_size, act_sizes_all, max_episodes=self.config.env.parallel["sp_collection"], max_eps_length=self.config.train["timesteps_per_update"]
                )

                if self.config.env.parallel["xp_collection"] != 0:
                    self.cross_play_exp_replay = EpisodicExperienceReplay(
                        tuple_obs_size, act_sizes_all, max_episodes=self.config.env.parallel["xp_collection"], max_eps_length=self.config.train["timesteps_per_update"]
                    )

            # Eval policy after sufficient number of episodes were collected.
            agent_population.save_model(ckpt_id+1, save_model=(self.logger.save_model
                                                               and ((ckpt_id+1) % self.logger.save_model_period == 0))
                                       )
            # Compute self-play and cross-play matrix. Save and add them to logs in logger.
            sp_mat = self.eval_sp_policy_performance(agent_population, self.logger, ckpt_id+1)
            xp_mat = self.eval_xp_policy_performance(agent_population, self.logger, ckpt_id+1)
            for a_id in range(self.config.populations["num_agents_per_population"]):
                self.logger.log_xp_matrix(f"Returns/xp_matrix{a_id}", sp_mat[:, :, a_id] + xp_mat[:, :, a_id], checkpoint=ckpt_id+1)
            self.logger.commit()

class Logger:
    """
        Class to initialize logger object for writing down experiment resulst to wandb.
    """
    def __init__(self, config):
        
        logger_period = config.logger.logger_period 
        self.save_model = config.logger.get("save_model", False)
        # For metrics that are not saved every checkpoint,
        # this deterHow many update steps before one logs the value
        self.save_model_period = config.logger.get("save_model_period", 20)
        if not "sp_collection" in config.env.parallel.keys() and not "xp_collection" in config.env.parallel.keys():
            self.steps_per_update = (config.env.parallel.agent1_collection + config.env.parallel.agent2_collection) * config.train.timesteps_per_update
        else:
            self.steps_per_update = (config.env.parallel.sp_collection + config.env.parallel.xp_collection) * config.train.timesteps_per_update
        if logger_period < 1:
            # Frequency
            self.train_log_period = int(logger_period * config.run.num_timesteps // self.steps_per_update)
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

    def log_xp_matrix(self, tag, mat, step=None, columns=None, rows=None, commit=False, **kwargs):
        if rows is None:
            rows = [str(i) for i in range(mat.shape[0])]
        if columns is None:
            columns = [str(i) for i in range(mat.shape[1])]
        tab = wandb.Table(
                columns=columns,
                data=mat,
                rows=rows
                )
        wandb.log({tag: tab, **kwargs}, step=step, commit=commit)

    def define_metrics(self):
        wandb.define_metric("train_step")
        wandb.define_metric("checkpoint")
        wandb.define_metric("Train/*", step_metric="train_step")
        wandb.define_metric("Returns/*", step_metric="checkpoint")
