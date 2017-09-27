import tensorflow as tf
import gym
import os
import math
import numpy as np
import random
# from gym import envs
#Define the saving path
Load_path = '../../Data/A.3/'
Model_path = '../../Model/A.3/Linear/'
Save_path = '../../result/A.3/Linear/'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)
#Model saving function
def save_model(session):
    if not os.path.exists(Model_path):
        os.makedirs(Model_path)
    saver = tf.train.Saver()
    saver.save(session, Model_path+'Linear_Model.checkpoint')
#Define the event
env = gym.make('CartPole-v1')
# env = gym.wrappers.Monitor(env, Evual_path+'cartpole_experiment_1')
#print out the action space and observation space
print(env.action_space)
print(env.observation_space)
#Read the data
Sample_sate = np.load(Load_path+'Sample_sate.npy')
Sample_action = np.load(Load_path+'Sample_action.npy')
Sample_reward = np.load(Load_path+'Sample_reward.npy')
Sample_sate_new = np.load(Load_path+'Sample_sate_new.npy')
print(Sample_action.shape)
# #Pre-process it into a list
# Samples = []
# for item in range(Sample_sate.shape[0]):
#     print('\rPre-processing...'+str(item/Sample_sate.shape[0]),end='')
#     Samples.append([Sample_sate[item],Sample_action[item],Sample_reward[item],Sample_sate_new[item]])

#Define global variables
#Reinforcement Learning Varaibles
n_episode = 100
episode_len = 300
Gamma = 0.99    #Discount factor
#Deep Learning Variables
batch_size = 64
nEpochs = 10
LR = 0.5

#Construct the neural network
#Defining place holders for tensor
x = tf.placeholder('float',[None,4])      #batch_size * Dim
y = tf.placeholder('float',[None,2])              #Batch_size * output number

seed_selected = round(random.uniform(1,5000))
print('The selected seed is:',seed_selected)

def neural_net_work(x):
    tf.set_random_seed(seed=4711)
    parameters = {'weight':tf.Variable(tf.random_normal([4,2])),\
                  'bias':tf.Variable(tf.random_normal([2]))}
    # output = tf.add(tf.matmul(x,parameters['weight']),parameters['bias'])
    output = tf.add(tf.matmul(x, parameters['weight']),parameters['bias'])

    return output

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

#Training
def batch_reinforcement_learning(x):
    prediction = neural_net_work(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    optimiser = tf.train.MomentumOptimizer(learning_rate=LR,momentum=0.1).minimize(loss)
    # optimiser = tf.train.RMSPropOptimizer(learning_rate=LR,decay=0.8,momentum=0.2,centered=True).minimize(loss)
    # define the variables to evaluate the performace
    run_eps_len_collection = []
    final_reward_collection = []
    Bellman_loss_collection = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for cEpoch in range(nEpochs):
            current_Epoch_Loss = 0
            n = Sample_sate.shape[0]
            random_index = random.sample(range(n), n)
            # for each epoch we need times to perform stochastic gradient descent
            for i in range(n // batch_size):
                current_x = Sample_sate[random_index[i * batch_size: (i + 1) * batch_size]]
                # print(current_x.shape)
                current_y = Sample_action[random_index[i * batch_size: (i + 1) * batch_size]]
                # print(current_y.shape)
                _, currentloss = sess.run([optimiser, loss], feed_dict={x: current_x, y: current_y})
                current_Epoch_Loss += currentloss
                # print(np.shape(current_Epoch_Loss))
            print(cEpoch + 1, 'iteration has been completed and the loss of this epoch is', current_Epoch_Loss)
            state_obs_feed_dict = {x: Sample_sate, y: np.zeros((n, 2))}
            state_new_feed_dict = {x: Sample_sate_new, y: np.zeros((n, 2))}
            action_decoded = np.argmin(Sample_action,axis=1)
            Q_value_states = prediction.eval(state_obs_feed_dict)[np.arange(action_decoded.shape[0]),action_decoded]
            Q_value_new_states = np.max(prediction.eval(state_new_feed_dict), axis=1)  # n_sample * 1
            # print(Sample_reward.shape)
            Bellman_losses = 0.5 * np.mean(np.square(np.subtract(np.add(Sample_reward, Gamma*Q_value_new_states), Q_value_states)))
            print('The Bellman loss of the', cEpoch + 1, 'iteration is:', Bellman_losses)
            Bellman_loss_collection.append(Bellman_losses)
            #Evulate the performance
            env.reset()  # Reset the initialization and start a new episode
            prev_stat_obs = []  # No initial observation
            for t in range(episode_len):
                # env.render()             #Display the gaming result
                if t == 0:
                    action = env.action_space.sample()  # Random action
                else:
                    input_state = np.reshape(prev_stat_obs, [1, 4])
                    action_dict = {x: input_state, y: np.zeros((1, 2))}
                    action = np.asscalar(np.argmax(prediction.eval(action_dict), axis=1))
                    # print(prediction.eval(action_dict))
                    # print(action)
                stat_obs, _, done, info = env.step(action)
                reward = reward_transform(done)
                run_eps_len = t + 1
                # action_encoded = one_hot_encode(action)
                if done == True:
                    print('Episode ', cEpoch + 1, ' terminated at ', t + 1, 'steps')
                    break
                elif t == (episode_len - 1):
                    print('Episode ', cEpoch + 1, 'does not terminate in 300 steps.')
                prev_stat_obs = stat_obs
            run_eps_len_collection.append(run_eps_len)
            final_reward = math.pow(Gamma, run_eps_len - 1) * (-1)
            final_reward_collection.append(final_reward)
        save_model(sess)
        np.save(Save_path+'Bellmanloss linear learning rate = '+str(LR),np.asarray(Bellman_loss_collection))
        np.save(Save_path+'Performance_length with LR='+str(LR),np.asarray(run_eps_len_collection))
        np.save(Save_path + 'discounted_reward with LR=' + str(LR), np.asarray(final_reward_collection))

        # Episode length mean and standard variance
        episode_len_mean = np.mean(np.asarray(run_eps_len_collection), axis=0)
        episode_len_std = np.std(np.asarray(run_eps_len_collection), axis=0)
        print('The mean of episode length is:', episode_len_mean, ' and the standard deviation is:', episode_len_std)
        print('The longest step length is:', np.max(np.asarray(run_eps_len_collection)))
        # Rewards mean and standard variance
        rewards_mean = np.mean(np.asarray(final_reward_collection), axis=0)
        rewards_std = np.std(np.asarray(final_reward_collection), axis=0)
        print('The mean of reward length is:', rewards_mean, ' and the standard deviation is:', rewards_std)


batch_reinforcement_learning(x)