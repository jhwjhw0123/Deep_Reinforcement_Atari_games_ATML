import tensorflow as tf
import gym
import os
import math
import numpy as np
import random
from operator import itemgetter
import matplotlib.pyplot as plt

# from gym import envs
#Define the saving path
Load_path = '../../Data/A.7/'
Model_path = '../../Model/A.7/Model/'
Save_path = '../../result/A.7/'
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

random_seed = round(random.uniform(1,50000))
print('seed seleted:',random_seed)

#Define global variables
#Reinforcement Learning Varaibles
n_episode = 2000
episode_len = 300
Gamma = 0.99    #Discount factor
Epsilon = 0.05
buffer_len = 2000
#Deep Learning Variables
batch_size = 128
LR = 1e-7

#Construct the neural network
#Defining place holders for tensor
s_current = tf.placeholder('float',[None,4])              #batch_size * Dim for the current state
s_next = tf.placeholder('float',[None,4])              #batch_size * Dim for the next state
y_action = tf.placeholder('float',[None,2])              #Batch_size * output number
r = tf.placeholder('float',[None,1])              #Batch_size * 1 (rewards)

def neural_net_work(s_current,s_next,r):
    # tf.set_random_seed(1783)
    tf.set_random_seed(1335)
    #Variables of the training neural networks
    Weights = {'hidden':tf.Variable(tf.random_normal([4,100])),\
               'output':tf.Variable(tf.random_normal([100,2]))}
    Bias = {'hidden':tf.Variable(tf.random_normal([100])),\
            'output':tf.Variable(tf.random_normal([2]))}
    #Variables of target neural networks
    Weights_target = {'hidden': tf.Variable(tf.random_normal([4, 100])), \
               'output': tf.Variable(tf.random_normal([100, 2]))}
    Bias_target = {'hidden': tf.Variable(tf.random_normal([100])), \
            'output': tf.Variable(tf.random_normal([2]))}

    # output = tf.add(tf.matmul(s_current,parameters['weight']),parameters['bias'])
    hidden_input_current = tf.add(tf.matmul(s_current,Weights['hidden']),Bias['hidden'])
    hidden_output_current = tf.nn.relu(hidden_input_current)
    output_current = tf.add(tf.matmul(hidden_output_current,Weights['output']),Bias['output'])

    # Process the next state with the same network
    hidden_input_next = tf.add(tf.matmul(s_next,Weights_target['hidden']),Bias_target['hidden'])
    hidden_output_next = tf.nn.relu(hidden_input_next)
    output_next = tf.add(tf.matmul(hidden_output_next,Weights_target['output']),Bias_target['output'])

    #copy the values
    global copy_weight_hidden_variable
    copy_weight_hidden_variable = Weights_target['hidden'].assign(Weights['hidden'])
    global copy_weight_output_variable
    copy_weight_output_variable = Weights_target['output'].assign(Weights['output'])
    global copy_bias_hidden_variable
    copy_bias_hidden_variable = Bias_target['hidden'].assign(Bias['hidden'])
    global copy_bias_output_variable
    copy_bias_output_variable = Bias_target['output'].assign(Bias['output'])

    return output_current,output_next,r

#Copy target Q-network function
def copy_target_network(session,copy_list):
    for copy_actions in copy_list:
        session.run(copy_actions)

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

loss_plot = []
episode_plot = []
dis_reward_plot = []

def online_train_neural_network(s_current,r):
    prediction_current,prediction_next,reward_tensor = neural_net_work(s_current,s_next,r)
    action_selected = tf.argmax(tf.nn.softmax(prediction_current),axis=1)
    action_selected = tf.cast(action_selected,tf.int32)
    # action_encoded = tf.cast(tf.contrib.layers.one_hot_encoding(action_selected,num_classes=2,on_value=0,off_value=1),tf.float32)
    #Learning directly from Bellman loss
    n_data = tf.shape(y_action)[0]     #Length
    rwo_list = tf.reshape(tf.range(0,limit=n_data),[n_data,1])
    y_action_encoded = tf.cast(tf.argmin(y_action,axis=1),tf.int32)
    indexing_list = tf.concat([rwo_list,tf.reshape(y_action_encoded,[n_data,1])],axis=1)
    Q_s_a = tf.gather_nd(prediction_current, indexing_list)
    ####################The next two lines are for test usage########################
    # Q_s_a_test = tf.gather_nd(prediction_next, indexing_list)
    # Q_s_a = tf.py_func(index_along_every_row, [prediction_current, action_selected], [tf.float32])[0]
    #/*******************************end test***************************************/
    max_Q_s_a_next = tf.reduce_max(prediction_next,axis=1)
    difference = reward_tensor + Gamma*tf.stop_gradient(max_Q_s_a_next) - Q_s_a
    loss = tf.reduce_mean(0.5*tf.square(difference))
    # optimiser = tf.train.MomentumOptimizer(learning_rate=LR,momentum=0.2).minimize(loss)
    # optimiser = tf.train.RMSPropOptimizer(learning_rate=LR,decay=0.9,momentum=0.5,centered=True).minimize(loss)
    optimiser = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

    #Difine the variables to collect the informations
    step_len_500 = []
    step_len_1000 = []
    step_len_1500 = []
    step_len_2000 = []

    #Define replay buffer
    replay_buffer = []
    #count the episode that we start to train
    train_start_episode = 0


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_list = []
        episode_list = []
        dis_reward_list = []
        copy_target_network(sess, [copy_weight_hidden_variable, copy_weight_output_variable,
                                  copy_bias_hidden_variable,
                                  copy_bias_output_variable])    #Initial copy of the network
        for c_episode in range(n_episode):
            # print_flag = False          #For test usage
            if train_start_episode!=0 and (c_episode - train_start_episode)%5 == 0:
                copy_target_network(sess, [copy_weight_hidden_variable, copy_weight_output_variable,
                                           copy_bias_hidden_variable,
                                           copy_bias_output_variable])
                print('Target Q-network updated')
            episode_loss = 0
            reward = 0
            run_eps_len = 0
            env.reset()      #Reset the initialization and start a new episode
            prev_stat_obs = []      #No initial observation
            for t in range(episode_len):
                # env.render()             #Display the gaming result
                if t==0:
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
                if len(replay_buffer) > batch_size:
                    #Sample from replay buffer
                    random_index = random.sample(range(len(replay_buffer)), batch_size)
                    sample_selected = itemgetter(*random_index)(replay_buffer)
                    learn_state_current_list = []
                    learn_state_next_list = []
                    learn_reward_list = []
                    learn_action_list = []
                    for learn_sample in sample_selected:
                        learn_state_current_list.append(learn_sample[0])
                        learn_state_next_list.append(learn_sample[3])
                        learn_reward_list.append([learn_sample[2]])
                        learn_action_list.append(np.array(one_hot_encode(learn_sample[1])))
                    learn_state_current = np.stack(learn_state_current_list,axis=0)
                    learn_state_next = np.stack(learn_state_next_list,axis=0)
                    learn_action = np.stack(learn_action_list,axis=0)
                    learn_reward = np.stack(learn_reward_list,axis=0)
                    learn_dict = {s_current:learn_state_current,s_next:learn_state_next,y_action:learn_action,r:learn_reward}
                    _, currentloss = sess.run([optimiser, loss], feed_dict=learn_dict)
                    episode_loss += currentloss
                    loss_list.append(currentloss)

                    if train_start_episode == 0:
                        train_start_episode = c_episode
                    # test_dict = {s_current: np.reshape(np.array(replay_buffer[0][3]), [1, 4]), s_next: np.reshape(replay_buffer[0][3], [1, 4]),
                    #              y_action: np.reshape(np.array(one_hot_encode(replay_buffer[13][1])),[1,2]), r: learn_reward}
                    # if print_flag == False:
                    #     print(y_action_encoded.eval(test_dict))
                    #     print(replay_buffer[13][1])
                    #     print_flag = True
                        # print(y_action_encoded.eval(test_dict))
                else:
                    pass
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
                    episode_list.append(run_eps_len)
                    this_dis_reward = math.pow(Gamma, run_eps_len - 1) * (-1)
                    dis_reward_list.append(this_dis_reward)

                    if c_episode % 20 == 0 and c_episode != 0:
                        loss_list = np.array(loss_list)
                        episode_list = np.array(episode_list)
                        loss_plot.append(np.mean(loss_list))
                        episode_plot.append(np.mean(episode_list))
                        dis_reward_plot.append(np.mean(dis_reward_list))
                        loss_list = []
                        dis_reward_list = []
                        episode_list = []
                    print('Episode ',c_episode+1,' terminated at ',t+1,'steps')
                    break
                elif t == (episode_len-1):
                    print('Episode ',c_episode+1,'does not terminate in 300 steps.')
                if t!= 0:
                    replay_buffer.append([prev_stat_obs,action,reward,stat_obs])
                if len(replay_buffer)>buffer_len:
                    replay_buffer.pop(0)
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
        save_model(sess)
        print('The average step-length of the first 500 episode is:', np.mean(np.asarray(step_len_500), axis=0))
        print('The average step-length of the 500 to 1000 episode is:', np.mean(np.asarray(step_len_1000), axis=0))
        print('The average step-length of the 1000 to 1500 episode is:', np.mean(np.asarray(step_len_1500), axis=0))
        print('The average step-length of the 1500 to 2000 episode is:', np.mean(np.asarray(step_len_2000), axis=0))

# online_train_neural_network(s_current,r)
# np.save(Save_path+'episode_performance',episode_plot)
# np.save(Save_path+'loss',loss_plot)
# np.save(Save_path+'reawrd',dis_reward_plot)
episode_plot = np.load(Save_path+'episode_performance'+'.npy')
loss_plot = np.load(Save_path+'loss'+'.npy')
dis_reward_plot = np.load(Save_path+'reawrd'+'.npy')
# plt.subplot(1,3,1)
# plt.plot(episode_plot)
# plt.xlabel('Episode length')
# plt.show()
# loss_plot = np.load(Save_path+'loss_30.npy')
# plt.subplot(1,3,2)
# plt.plot(loss_plot)
# plt.xlabel('loss function')
# plt.subplot(1,3,3)
plt.plot(dis_reward_plot)
plt.xlabel('discounted rewards')
plt.show()
###################Testing Codes######################
# aa = tf.constant([[8,5],[7,4],[1,2],[4,6]])
# bb = tf.reshape(tf.constant([0,1,1,0]),[4,1])
# ee = tf.contrib.layers.one_hot_encoding(bb,num_classes=2,on_value=0,off_value=1)
# rwo_list = tf.reshape(tf.range(0,limit=4),[4,1])
# indexing_list = tf.concat([rwo_list,bb],axis=1)
# ff = tf.gather_nd(aa,indexing_list)
# with tf.Session() as sess:
#     print(ff.eval())

