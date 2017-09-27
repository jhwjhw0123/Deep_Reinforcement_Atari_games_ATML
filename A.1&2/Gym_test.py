import gym
import os
import math
import numpy as np
# from gym import envs
#Define the saving path
Save_path = '../../result/A.2/'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)
#Define the event
env = gym.make('CartPole-v0')
# env = gym.wrappers.Monitor(env, Evual_path+'cartpole_experiment_1')
#print out the action space and observation space
print(env.action_space)
print(env.observation_space)
#Define global variables
n_episode = 100
episode_len = 300
Gamma = 0.99    #Discount factor
#List variables to collect results
run_eps_len_collection = []
final_reward_collection = []

for c_episode in range(n_episode):
    env.reset()      #Reset the initialization and start a new episode
    for t in range(episode_len):
        env.render()             #Display the gaming result
        action = env.action_space.sample()   #Random action
        obs,_,done,info = env.step(action)
        if done == True:
            reward = -1
            print('Episode ',c_episode+1,' terminated at ',t+1,'steps')
            run_eps_len = t+1
            break
        elif t == (episode_len-1):
            reward = 0
            run_eps_len = 300
            print('Episode ',c_episode+1,'does not terminate in 300 steps.')
        else:
            reward = 0
    run_eps_len_collection.append(run_eps_len)
    final_reward = math.pow(Gamma,run_eps_len-1)*(-1)
    final_reward_collection.append(final_reward)
    print('The reward of this episode is:',final_reward)

#Episode length mean and standard variance
episode_len_mean = np.mean(np.asarray(run_eps_len_collection),axis=0)
episode_len_std = np.std(np.asarray(run_eps_len_collection),axis=0)
print('The mean of episode length is:',episode_len_mean,' and the standard deviation is:',episode_len_std)
#Rewards mean and standard variance
rewards_mean = np.mean(np.asarray(final_reward_collection),axis=0)
rewards_std = np.std(np.asarray(final_reward_collection),axis=0)
print('The mean of reward length is:',rewards_mean,' and the standard deviation is:',rewards_std)

# np.save(Save_path+'episode_length_mean',episode_len_mean)
# np.save(Save_path+'episode_length_std',episode_len_std)
# np.save(Save_path+'rewards_mean',rewards_mean)
# np.save(Save_path+'rewards_std',rewards_std)