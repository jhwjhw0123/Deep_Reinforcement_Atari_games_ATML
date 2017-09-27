import tensorflow as tf
import gym
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
# from gym import envs
#Define the saving path
Load_path = '../../Data/A.4/'
Model_path = '../../Model/A.4/Model/'
Save_path = '../../result/A.4/'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)
#Model saving function
def save_model(session):
    if not os.path.exists(Model_path):
        os.makedirs(Model_path)
    saver = tf.train.Saver()
    saver.save(session, Model_path+'Onlinelearing_Model.checkpoint')
#Define the event
env = gym.make('CartPole-v1')
# env = gym.wrappers.Monitor(env, Evual_path+'cartpole_experiment_1')
#print out the action space and observation space
print(env.action_space)
print(env.observation_space)

#Define global variables
#Reinforcement Learning Varaibles
n_episode = 2000
episode_len = 300
Gamma = 0.99    #Discount factor
Epsilon = 0.05
#Deep Learning Variables
nEpochs = 200
LR = 1e-6

#Construct the neural network
#Defining place holders for tensor
s_current = tf.placeholder('float',[None,4])              #batch_size * Dim for the current state
s_next = tf.placeholder('float',[None,4])              #batch_size * Dim for the next state
y_action = tf.placeholder('float',[None,2])              #Batch_size * output number
r = tf.placeholder('float',[None,1])              #Batch_size * 1 (rewards)


def neural_net_work(s_current,s_next,r):
    # tf.set_random_seed(2017)
    Weights = {'hidden':tf.Variable(tf.random_normal([4,100])),\
               'output':tf.Variable(tf.random_normal([100,2]))}
    Bias = {'hidden':tf.Variable(tf.random_normal([100])),\
            'output':tf.Variable(tf.random_normal([2]))}
    # output = tf.add(tf.matmul(s_current,parameters['weight']),parameters['bias'])
    hidden_input_current = tf.add(tf.matmul(s_current,Weights['hidden']),Bias['hidden'])
    hidden_output_current = tf.nn.relu(hidden_input_current)
    output_current = tf.add(tf.matmul(hidden_output_current,Weights['output']),Bias['output'])

    # Process the next state with the same network
    hidden_input_next = tf.add(tf.matmul(s_next,Weights['hidden']),Bias['hidden'])
    hidden_output_next = tf.nn.relu(hidden_input_next)
    output_next = tf.add(tf.matmul(hidden_output_next,Weights['output']),Bias['output'])

    return output_current,output_next,r

#Action Encoding
def one_hot_encode(action):
    if action == 0:
        return [0,1]
    elif action==1:
        return [1,0]
    else:
        raise ValueError('Action could only be 0 or 1')

#Reward encoding
def reward_transform(done_mark):
    if done_mark == True:
        return -1
    elif done_mark == False:
        return 0
    else:
        raise ValueError('done mark must be logic True or False')

#Tensor row-wise indexing
# def index_along_every_row(array, index):
#     N, _ = array.shape
#     return array[np.arange(N), index]

episode_plot = np.zeros(2000)

def online_train_neural_network(s_current,r):
    prediction_current,prediction_next,reward_tensor = neural_net_work(s_current,s_next,r)
    action_selected = tf.argmax(tf.nn.softmax(prediction_current),axis=1)
    action_selected = tf.cast(action_selected,tf.int32)
    # action_encoded = tf.cast(tf.contrib.layers.one_hot_encoding(action_selected,num_classes=2,on_value=0,off_value=1),tf.float32)
    #Learning directly from Bellman loss
    n_data = tf.shape(y_action)[0]     #Length
    rwo_list = tf.reshape(tf.range(0,limit=n_data),[n_data,1])
    y_action_encoded = tf.cast(tf.argmin(y_action, axis=1), tf.int32)
    indexing_list = tf.concat([rwo_list, tf.reshape(y_action_encoded, [n_data, 1])], axis=1)
    Q_s_a = tf.gather_nd(prediction_current, indexing_list)
    # Q_s_a = tf.py_func(index_along_every_row, [prediction_current, action_selected], [tf.float32])[0]
    max_Q_s_a_next = tf.reduce_max(prediction_next,axis=1)
    difference = reward_tensor + Gamma*tf.stop_gradient(max_Q_s_a_next) - Q_s_a
    loss = tf.reduce_mean(0.5*tf.square(difference))
    optimiser = tf.train.MomentumOptimizer(learning_rate=LR,momentum=0.1).minimize(loss)
    # optimiser = tf.train.RMSPropOptimizer(learning_rate=LR,decay=0.8,momentum=0.2,centered=True).minimize(loss)
    # optimiser = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)


    #Difine the variables to collect the informations
    step_len_500 = []
    step_len_1000 = []
    step_len_1500 = []
    step_len_2000 = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for c_episode in range(n_episode):
            episode_loss = 0
            reward = 0
            run_eps_len = 0
            env.reset()      #Reset the initialization and start a new episode
            prev_stat_obs = []      #No initial observation
            for t in range(episode_len):
                # env.render()             #Display the gaming result
                if t == 0:
                    action = env.action_space.sample()   #Random action
                    stat_obs, _, done, info = env.step(action)
                else:
                    input_state_current = np.reshape(prev_stat_obs,[1,4])
                    current_reward = np.asarray([[reward]])
                    action_dict = {s_current:input_state_current,s_next:input_state_current,y_action:np.zeros((1,2)),r:current_reward}
                    #Using rejection sampling to get a random number
                    uni_sample = random.uniform(0,1)
                    if uni_sample>Epsilon:
                        action = np.asscalar(action_selected.eval(action_dict))   # Exploit the best action
                    else:
                        action = env.action_space.sample()  # Random action
                    # print(prediction_current.eval(action_dict))
                    # print(action)
                    stat_obs, _, done, info = env.step(action)
                    input_state_next = np.reshape(stat_obs,[1,4])
                    action_input = np.reshape(np.array(one_hot_encode(action)),[1,2])
                    learn_dict = {s_current:input_state_current,s_next:input_state_next,y_action:action_input,r:current_reward}
                    _, currentloss = sess.run([optimiser, loss], feed_dict=learn_dict)
                    episode_loss += currentloss

                    # print(Q_s_a.eval(learn_dict))
                    # print(prediction_current.eval(learn_dict))
                    # print(prediction_next.eval(learn_dict))
                    # print(max_Q_s_a_next.eval(learn_dict))
                    # print(difference.eval(learn_dict))
                    # print(loss.eval(learn_dict))
                    # print('Go to next time step\n\n')
                reward = reward_transform(done)
                run_eps_len = t + 1
                # action_encoded = one_hot_encode(action)
                if done == True:
                    episode_plot[c_episode] += run_eps_len
                    print('Episode ',c_episode+1,' terminated at ',t+1,'steps')
                    break
                elif t == (episode_len-1):
                    print('Episode ',c_episode+1,'does not terminate in 300 steps.')
                prev_stat_obs = stat_obs
            if c_episode<500:
                step_len_500.append(run_eps_len)
            elif c_episode<1000:
                step_len_1000.append(run_eps_len)
            elif c_episode<1500:
                step_len_1500.append(run_eps_len)
            else:
                step_len_2000.append(run_eps_len)
            print('The loss of ',c_episode+1,'episode is:',episode_loss)
        # save_model(sess)
        print('The average step-length of the first 500 episode is:', np.mean(np.asarray(step_len_500), axis=0))
        print('The average step-length of the 500 to 1000 episode is:', np.mean(np.asarray(step_len_1000), axis=0))
        print('The average step-length of the 1000 to 1500 episode is:', np.mean(np.asarray(step_len_1500), axis=0))
        print('The average step-length of the 1500 to 2000 episode is:', np.mean(np.asarray(step_len_2000), axis=0))

n_trail = 3
for i in range(n_trail):
    print('Processing the',i,'iteration')
    online_train_neural_network(s_current,r)
# # episode_plot = episode_plot/n_trail
# # np.save(Save_path+'episode_performance',episode_plot)
# episode_plot = np.load(Save_path+'episode_performance.npy')
# plt.plot(episode_plot)
# plt.show()

###################Testing Codes######################
# aa = tf.constant([[8,5],[7,4],[1,2],[4,6]])
# bb = tf.reshape(tf.constant([0,1,1,0]),[4,1])
# ee = tf.contrib.layers.one_hot_encoding(bb,num_classes=2,on_value=0,off_value=1)
# rwo_list = tf.reshape(tf.range(0,limit=4),[4,1])
# indexing_list = tf.concat([rwo_list,bb],axis=1)
# ff = tf.gather_nd(aa,indexing_list)
# with tf.Session() as sess:
#     print(ff.eval())

