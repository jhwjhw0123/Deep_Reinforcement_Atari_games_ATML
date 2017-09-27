import gym
import os
import math
import numpy as np
# from gym import envs
#Define the saving path
Save_path = '../../Data/A.3/'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)
#Define the event
env = gym.make('CartPole-v1')
# env = gym.wrappers.Monitor(env, Evual_path+'cartpole_experiment_1')
#print out the action space and observation space
print(env.action_space)
print(env.observation_space)
#Define global variables
#Reinforcement Learning Varaibles
n_episode = 200000
episode_len = 300
Gamma = 0.99    #Discount factor
#List variables to collect resultss
Sample_sate_collection = []
Sample_action_collection = []
Sample_reward_collection = []
Sample_sate_new_collection = []
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
length_clolection = []
#Sampling
n_valid_episode = 0
for c_episode in range(n_episode):
    env.reset()      #Reset the initialization and start a new episode
    prev_stat_obs = []      #No initial observation
    temp_sate_collection = []
    temp__action_collection = []

    for t in range(episode_len):
        # env.render()             #Display the gaming result
        action = env.action_space.sample()   #Random action
        stat_obs,_,done,info = env.step(action)
        reward = reward_transform(done)
        run_eps_len = t + 1
        delete_len = run_eps_len
        action_encoded = one_hot_encode(action)
        try:
            if prev_stat_obs != []:
                Sample_sate_collection.append(prev_stat_obs)
                Sample_action_collection.append(action_encoded)
                Sample_reward_collection.append(reward)
                Sample_sate_new_collection.append(stat_obs)
        except:
            delete_len = delete_len - 1
            pass
        if done == True:
            print('Episode ',c_episode+1,' terminated at ',t+1,'steps')
            #Delete some not very good example
            if run_eps_len>50:
                del Sample_sate_collection[-3:]
                del Sample_action_collection[-3:]
                del Sample_reward_collection[-3:]
                del Sample_sate_new_collection[-3:]
                n_valid_episode += 1
            else:
                del Sample_sate_collection[-delete_len:]
                del Sample_action_collection[-delete_len:]
                del Sample_reward_collection[-delete_len:]
                del Sample_sate_new_collection[-delete_len:]
            length_clolection.append(t+1)
            break
        elif t == (episode_len-1):
            print('Episode ',c_episode+1,'does not terminate in 300 steps.')
        prev_stat_obs = stat_obs
    if n_valid_episode>=2000:
        print('Actually runed',c_episode+1,'episodes to collect the data')
        break

print('Average length is:',np.mean(np.asarray(length_clolection)))
print('Maximum length is:',np.max(np.asarray(length_clolection)))
#Convert to numpy
Sample_sate = np.stack(Sample_sate_collection,axis=0)
Sample_action = np.stack(Sample_action_collection,axis=0)
Sample_reward = np.stack(Sample_reward_collection,axis=0)
Sample_sate_new = np.stack(Sample_sate_new_collection,axis=0)

#Save the data
np.save(Save_path+'Sample_sate',Sample_sate)
np.save(Save_path+'Sample_action',Sample_action)
np.save(Save_path+'Sample_reward',Sample_reward)
np.save(Save_path+'Sample_sate_new',Sample_sate_new)