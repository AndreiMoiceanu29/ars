# Ai 2019
# Importing the libs
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
# Setting the Hyper Parameters
class Hp():
    
    def __init__(self):
        self.num_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.num_directions = 16
        self.num_best_directions = 16
        assert self.num_best_directions <= self.num_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = 'HalfCheetahBulletEnv-v0'
    
        
# Normalizing the states
        
class Normalizer():
    
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.variance = np.zeros(num_inputs)
    
    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff +=(x - last_mean) * (x - self.mean)
        self.variance = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_stddev = np.sqrt(self.variance)
        return (inputs - obs_mean) / obs_stddev

# Buliding the AI

class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))
    
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise * delta).dot(input)
    
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.num_directions)]
    
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate/(hp.num_best_direction * sigma_r)

# Exploring the policy on one specific direction over one episode
def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1) #Classic trick in RL
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

# Training the AI
def train(env, policy, normalizer, hp):
    
    for step in range(hp.num_steps):
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.num_directions
        negative_rewards = [0] * hp.num_directions
        
        # Getting the positive rewards in positive directions
        for k in range(hp.num_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = 'positive', delta = deltas[k])
        
        # Gettint the negative rewards in negative directions
        for k in range(hp.num_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = 'negative', delta = deltas[k])
        
        # Gathering all the positive/negative rewards to compute the std deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r =  all_rewards.std()
        
        # Sorting the rollouts by the max(reward_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x])[:hp.num_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Update our policy
        policy.update(rollouts, sigma_r)
        
        # Printing the final reward of the policy after the update
        reward_eval = explore(env, normalizer, policy)
        print("Step: ",step,' Reward: ',reward_eval)

# Running the main code
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp','brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force = True)
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
policy = Policy(num_inputs, num_outputs)
normalizer = Normalizer(num_inputs)
train(env, policy, normalizer, hp)