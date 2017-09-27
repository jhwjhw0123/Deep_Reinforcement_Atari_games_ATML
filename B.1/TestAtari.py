import gym
import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# from gym import envs
#Define the saving path
Save_path = '../../result/B.1/Pong/'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)
#Define the event
# env = gym.make('Pong-v3')
# env = gym.make('MsPacman-v3')
env = gym.make('Boxing-v3')
print(env.action_space)
print(env.observation_space)
#Define global variables
n_episode = 100
episode_len = 30000
Gamma = 0.99    #Discount factor
img_size = 28

#frame pre-processing function
def Frame_pre_processing(image_data,Pong_Flag=False):
    img = Image.fromarray(image_data, 'RGB').convert('L')
    img = img.resize((img_size,img_size),resample=Image.BILINEAR)
    img_array = np.asarray(img, dtype=np.uint8)
    if Pong_Flag == True:
        img_array.setflags(write=True)
        img_array[img_array>=92] = 255

    return img_array

#Reward encoding
def reward_transform(initial_reward):
    if initial_reward >=1:
        return 1
    elif initial_reward <=-1:
        return -1
    else:
        return 0

Frame_count_collection = []
Game_score_collection = []
Discounted_score_collection = []
for c_episode in range(n_episode):
    env.reset()
    total_score = 0
    step_len = 0
    discounted_score = 0
    temp_image_buffer = []
    for t in range(episode_len):
        # env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        grey_obs = Frame_pre_processing(obs,Pong_Flag=True)
        # if t == 121:
        #     #Test Codes: Observe the image#
        #     fig = plt.figure(figsize=(6, 3.2))
        #     ax = fig.add_subplot(111)
        #     plt.imshow(grey_obs)
        #     print(grey_obs)
        #     plt.show()
            ####################
        total_score += reward
        discounted_score += math.pow(Gamma,t)*reward
        step_len = t+1
        if done == True:
            print('The agent terminated at',step_len,'step at the',c_episode+1,'episode')
            break
        elif t == episode_len-1:
            print('The agent didn\'t terminate after full steps at the',c_episode+1,'episode')
    Frame_count_collection.append(step_len)
    Game_score_collection.append(total_score)
    Discounted_score_collection.append(discounted_score)
    print('The score of the episode',c_episode+1,'is:', total_score)

#Episode length mean and standard variance
frame_count_mean = np.mean(np.asarray(Frame_count_collection),axis=0)
frame_count_std = np.std(np.asarray(Frame_count_collection),axis=0)
print('The mean of frame count is:',frame_count_mean,' and the standard deviation is:',frame_count_std)
#Scores mean and standard variance
score_mean = np.mean(np.asarray(Game_score_collection),axis=0)
score_std = np.std(np.asarray(Game_score_collection),axis=0)
print('The mean of game score is:',score_mean,' and the standard deviation is:',score_std)
#Discounted score
#Scores mean and standard variance
score_mean = np.mean(np.asarray(Discounted_score_collection),axis=0)
score_std = np.std(np.asarray(Discounted_score_collection),axis=0)
print('The mean of discounted score is:',score_mean,' and the standard deviation is:',score_std)

#Save the variables
# np.save(Save_path+'frame_count_mean',frame_count_mean)
# np.save(Save_path+'frame_count_std',frame_count_std)
# np.save(Save_path+'score_mean',score_mean)
# np.save(Save_path+'score_std',score_std)