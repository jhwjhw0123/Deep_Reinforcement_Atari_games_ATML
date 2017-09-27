import tensorflow as tf
import gym
import os
import math
import numpy as np
# from gym import envs
#Define the saving path
Save_path = '../../result/A.3/'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)
#Define the event
env = gym.make('CartPole-v0')
# env = gym.wrappers.Monitor(env, Evual_path+'cartpole_experiment_1')
#print out the action space and observation space
print(env.action_space)
print(env.observation_space)
#Define global variables
#Reinforcement Learning Varaibles
n_episode = 2000
episode_len = 300
Gamma = 0.99    #Discount factor
#Deep Learning Variables
batch_size = 128
nEpochs = 32
#Construct the neural network
x = tf.placeholder('float32',[None,4])  #Batch_size * Dim(4)
y = tf.placeholder('float32',[None,2])  #Batch_size * output(2 outputs)
#List variables to collect resultss
Sample_collection = []

#Reward tranformation
def reward_transform(done_mark):
    if done_mark == True:
        return -1
    elif done_mark == False:
        return 0
    else:
        raise ValueError('done mark must be logic True or False')

#define the function to encode actions into [0,1]
def one_hot_encode(action):
    if action == 0:
        return [0,1]
    elif action==1:
        return [1,0]
    else:
        raise ValueError('Action could only be 0 or 1')

#Sampling
for c_episode in range(n_episode):
    env.reset()      #Reset the initialization and start a new episode
    prev_stat_obs = []      #No initial observation
    for t in range(episode_len):
        # env.render()             #Display the gaming result
        action = env.action_space.sample()   #Random action
        stat_obs,_,done,info = env.step(action)
        reward = reward_transform(done)
        run_eps_len = t + 1
        action_encoded = one_hot_encode(action)
        try:
            Sample_collection.append([prev_stat_obs, action_encoded, reward, stat_obs])
        except:
            pass
        if done == True:
            print('Episode ',c_episode+1,' terminated at ',t+1,'steps')
            break
        elif t == (episode_len-1):
            print('Episode ',c_episode+1,'does not terminate in 300 steps.')
        prev_stat_obs = stat_obs

print(Sample_collection[1])