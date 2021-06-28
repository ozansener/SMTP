"""
Modified from https://github.com/modestyachts/ARS/blob/master/code/ars.py
"""
import time
import datetime
import gym
import numpy as np
from ars import ARSAggregator, ARSOptimizer
from shared_noise import *
from policies import *
import utils
import optimizers
from tensorboardX import SummaryWriter
from lqr_kakade import LQRKakade
import torch

@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed, params = None, deltas = None):

        self.env = gym.make(params['env_name'])
        self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.rollout_length = params['rollout_length']
        self.delta_std = params['delta_std']

        self.params = params

        if params['policy_type'] == 'linear':
            self.policy = LinearPolicy(params["policy_params"])
        else:
            raise NotImplementedError

        self.deltas = SharedNoiseTable(deltas, env_seed + 42) 
        
        self.is_adaptive_step_size = False #params['adaptive_step_size']

        self.momentum = params["momentum"]

    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.params["rollout_length"]

        total_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
            
        return total_reward, steps    

    def do_evaluate(self, w_policy, num_rollouts):
        rollout_rewards = []

        # set to false so that evaluation rollouts are not used for updating state statistics
        self.policy.update_filter = False
        self.policy.update_weights(w_policy)
            
        for i in range(num_rollouts):
            # for evaluation we do not shift the rewards (shift = 0) and we use the
            # default rollout length (1000 for the MuJoCo locomotion tasks)
            reward, step  = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
            rollout_rewards.append(reward)
                            
        return {'rollout_rewards': rollout_rewards}
 

    def do_rollouts(self, w_policy, velocity, z_iterate, num_rollouts = 1, iteration_id = 1, delta_idx = -1):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx, direction_weights = [], [], []
        steps = 0

        delta = self.deltas.get(delta_idx, w_policy.size)
        if self.is_adaptive_step_size:
            raise ValueError("Adaptive is not implemented")
            #delta = ((1.0/np.sqrt(iteration_id))*self.params["delta_std"] * delta).reshape(w_policy.shape)
        else:
            if self.momentum:
                pos_velocity = self.params["beta"] * velocity + delta.reshape(w_policy.shape)
                neg_velocity = self.params["beta"] * velocity - delta.reshape(w_policy.shape)

                delta_pos = self.params["step_size"] * (1.0 / (1.0 - self.params["beta"])) * pos_velocity
                delta_neg = self.params["step_size"] * (1.0 / (1.0 - self.params["beta"])) * neg_velocity
            else:
                delta_pos = (-1.0) * (self.params["delta_std"] * delta).reshape(w_policy.shape)
                delta_neg = (self.params["delta_std"] * delta).reshape(w_policy.shape)       

                z_iterate = w_policy
        deltas_idx.append(delta_idx)

        for i in range(num_rollouts):
            # set to true so that state statistics are updated 
            self.policy.update_filter = True

            # compute reward and number of timesteps used for positive perturbation rollout
            self.policy.update_weights(w_policy - delta_pos)
            pos_reward, pos_steps  = self.rollout(shift = self.params["shift"])
            
            # compute reward and number of timesteps used for negative pertubation rollout
            self.policy.update_weights(w_policy - delta_neg)
            neg_reward, neg_steps = self.rollout(shift = self.params["shift"]) 

            self.policy.update_weights(z_iterate)
            orig_reward, orig_steps = self.rollout(shift = self.params["shift"]) 

            steps += pos_steps + neg_steps

            rollout_rewards.append([pos_reward, neg_reward, orig_reward])
            direction_weights.append(1.0)
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps, "weights": direction_weights}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    def get_weights_plus_stats(self):
        return self.policy.get_weights_plus_stats()

class DFOLearner(object):
    """
    Derivative Free Optimization generic implementation
    """
    def __init__(self, params):
        self.aggregator = ARSAggregator(params)

        print(params)
        self.is_adaptive_step_size = False #params['adaptive_step_size']

        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = params["seed"] + 7)

        self.policy = LinearPolicy(params["policy_params"])
        self.policy_weights = self.policy.get_weights()

        self.z_iterate = np.copy(self.policy_weights)
        self.velocity = np.zeros(self.policy_weights.shape)

        self.momentum = params["momentum"]
        self.sampler = params["sampler"]
        if self.sampler == 'value' or self.sampler == 'angle':
            self.model = torch.load("{}_model".format(params["env_name"]), map_location='cpu')

 
        self.params = params
        self.workers = [Worker.remote(params["seed"] + 42 * i,
                                      params = params,
                                      deltas=deltas_id) for i in range(params["num_workers"])]
        self.timesteps = 0

        valid_keys =  [ "env_name", "seed", "delta_std", "step_size", "num_rollouts", "shift", "sampler", "momentum", "beta",  "adaptive_step_size"]
        exp_identifier = '___'.join(['{}_{}'.format(key,val) for (key,val) in params.items() if key in valid_keys])
        exp_identifier = "".join(exp_identifier.split())

        run_name = 'runs/{}_{}'.format(exp_identifier, datetime.datetime.now().strftime("%I%M%p_on_%B_%d_%Y"))
        run_name = "".join(run_name.split())
        self.writer = SummaryWriter(log_dir=run_name)

    def train(self, n_iter):
        train_start_time = time.time()
        for i in range(n_iter):
            iteration_id = i + 1

            step_start_time = time.time()
               
            delta_idx = self.sample_directions(self.policy_weights.size, iteration_id)
            deltas, rewards, weights = self.parallel_rollouts(iter_id = iteration_id, delta_id = delta_idx)

            if self.is_adaptive_step_size:
                mult = 1.0 / (np.sqrt(iteration_id))
            else:
                mult = 1.0
            
            self.compute_update(deltas, rewards, weights, mult)

            current_time = time.time()
            # Implement the filter updates

            for j in range(self.params['num_workers']):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
                break

            self.policy.observation_filter.stats_increment()
            self.policy.observation_filter.clear_buffer()

            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)

            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)            


            if (i + 1) % 5 == 0:
                eval_result = self.evaluate()
                print("Evaluation at step {} # step time:{} # total spent time:{}".format(i, 
                                                                                          current_time-step_start_time,
                                                                                          current_time-train_start_time))
                print("\t Average Reward:{} Reward Std:{}".format(eval_result["AverageRewards"], eval_result["StdRewards"]))
                print("\t Min Reward:{} Max Reward:{}".format(eval_result["MinRewards"], eval_result["MaxRewards"]))
                #w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                #np.savez("./logs/lin_policy_plus_at_{}_for_{}_reward_{}".format(i,self.params['env_name'], eval_result["AverageRewards"]), w)
                
                self.writer.add_scalar('average_reward', eval_result["AverageRewards"], self.timesteps)
                self.writer.add_scalar('std_reward', eval_result["StdRewards"], self.timesteps)
                self.writer.add_scalar('epoch', i, self.timesteps)

        #np.savez("gradients_and_more_for_{}".format(self.params["env_name"]), x=self.input_data, y=self.output_grad,
        #                                                                      mu=self.mu_log, std=self.std_log, 
        #                                                                      rd=self.saved_rollout_directions, rr=self.saved_rollout_rewards)
 

    def evaluate(self):
        policy_id = ray.put(self.policy_weights)
        rollout_per_worker = int(100/self.params['num_workers']) + 1

        rollouts = [ worker.do_evaluate.remote(policy_id, num_rollouts=rollout_per_worker) for worker in self.workers]

        results = ray.get(rollouts)

        rewards = []
        for result in results:
            rewards += result["rollout_rewards"]

        rewards = np.array(rewards, dtype=np.float64)
        return {'AverageRewards': np.mean(rewards), 'StdRewards': np.std(rewards), 
                'MaxRewards': np.max(rewards), 'MinRewards': np.min(rewards)}      

    def parallel_rollouts(self, iter_id=0, delta_id = 0):
        # Share current policy with workers
        policy_id = ray.put(self.policy_weights)
        if self.momentum:
            velocity_id = ray.put(self.velocity)
            z_iterate_id = ray.put(self.z_iterate)
        else:
            velocity_id = policy_id
            z_iterate_id = policy_id
        rollout_per_worker = int(self.params['num_rollouts']/self.params['num_workers'])

        # We assume number of rollouts is divisible to number of workers
        assert(self.params['num_workers']*rollout_per_worker == self.params['num_rollouts'])

        rollouts = [ worker.do_rollouts.remote(policy_id, velocity_id, z_iterate_id,
                                               num_rollouts=rollout_per_worker, 
                                               iteration_id=iter_id, 
                                               delta_idx=delta_id) for worker in self.workers]
        results = ray.get(rollouts)

        rewards, deltas, weights = [], [], []
        for result in results:
            self.timesteps += result["steps"] / float(self.params["rollout_length"])
            deltas += result["deltas_idx"]
            rewards += result["rollout_rewards"]
            weights += result["weights"]
        deltas = np.array(deltas)
        rewards = np.array(rewards, dtype=np.float64)
        weights = np.array(weights, dtype=np.float64)
        return deltas, rewards, weights

    def compute_update(self, deltasx, rewards, weights, mult):
        p = np.mean(rewards[:,0])
        n = np.mean(rewards[:,1])
        o = np.mean(rewards[:,2])

        if p>n and p>=o:
            # Positive step z_{+}
            if self.momentum:
                pos_velocity = self.params["beta"]*self.velocity + self.deltas.get(deltasx[0], self.policy_weights.size).reshape(self.policy_weights.shape)
                update = (-1.0)*self.params["step_size"]*pos_velocity
                self.velocity = pos_velocity
                self.z_iterate = self.policy_weights - self.params["step_size"] * (1.0 / (1.0 - self.params["beta"])) * pos_velocity
            else:
                update = self.deltas.get(deltasx[0], self.policy_weights.size).reshape(self.policy_weights.shape)*self.params["step_size"]*mult
            self.policy_weights += update            
        elif n>p and n>=o:
            # Negative step z_{-}
            if self.momentum:
                neg_velocity = self.params["beta"]*self.velocity - self.deltas.get(deltasx[0], self.policy_weights.size).reshape(self.policy_weights.shape)
                update = (-1.0)*self.params["step_size"]*neg_velocity
                self.velocity = neg_velocity
                self.z_iterate = self.policy_weights - self.params["step_size"] * (1.0 / (1.0 - self.params["beta"])) * neg_velocity
            else:
                update = (-1.0)*self.deltas.get(deltasx[0], self.policy_weights.size).reshape(self.policy_weights.shape)*self.params["step_size"]*mult
            self.policy_weights += update            
        return 0

    def get_right_input(self):
        w = ray.get(self.workers[0].get_weights_plus_stats.remote())
        pw = w[0]
        mu = w[1]
        std = np.copy(w[2])
        std[std<1e-7] = 1
        return pw, mu, std

    def sample_directions(self, direction_size, iter_id):
        if self.sampler == 'unit_normal':
            idx, delta = self.deltas.get_delta(direction_size)
            return idx
        else:
            raise ValueError("unly unit normal is supported")
