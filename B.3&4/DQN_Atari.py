import tensorflow as tf
import gym
import os
import math
import numpy as np
import random
from operator import itemgetter
from PIL import Image

# from gym import envs
#Define the saving path
Load_path = '../../Data/B.3&4/'
Model_path = '../../Model/B.3/Boxing/'
Save_path = '../../result/B.3/Boxing/'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)
#Model saving function
def save_model(session):
    if not os.path.exists(Model_path):
        os.makedirs(Model_path)
    saver = tf.train.Saver()
    saver.save(session, Model_path+'CNN_Model.checkpoint')

#Define the event
# env = gym.make('Pong-v3')
# env = gym.make('MsPacman-v3')
env = gym.make('Boxing-v3')
#print out the action space and observation space
print(env.action_space)
print(env.observation_space)

#Define global variables
#Reinforcement Learning Varaibles
n_episode = 1000000
episode_len = 30000
Gamma = 0.99    #Discount factor
Epsilon = 0.1
buffer_len = 150000
frame_dim = 4
action_space_size = 9
#Deep Learning Variables
batch_size = 64
LR = 1e-3
img_size = 28
#Convolutional channel
n_channel_layer1 = 16
n_channel_layer2 = 32
#Convolutional size
filter_size_layer1 = 6
filter_size_layer2 = 4

#Construct the neural network
#Defining place holders for tensor
s_current = tf.placeholder('float',[None,img_size,img_size,frame_dim])              #batch_size * Dim for the current state
s_next = tf.placeholder('float',[None,img_size,img_size,frame_dim])              #batch_size * Dim for the next state
y_action = tf.placeholder('float',[None,action_space_size])              #Batch_size * output number
r = tf.placeholder('float',[None,1])              #Batch_size * 1 (rewards)

def convolution2D(Data,Weight):
    return tf.nn.conv2d(Data,Weight,strides = [1,2,2,1], padding = 'SAME')

def maxpooling(Data):
    return tf.nn.max_pool(Data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
    # 'ksize' controls the size of the pooling window, 'strides' controls how it moves


def convolutional_neural_network(s_current,s_next,r):
    # tf.set_random_seed(1783)
    # tf.set_random_seed(2017)
    #Variables of the training neural networks
    Weights = {'conv_layer1':tf.Variable(tf.random_normal([filter_size_layer1,filter_size_layer1,frame_dim,n_channel_layer1])),\
               'conv_layer2':tf.Variable(tf.random_normal([filter_size_layer2,filter_size_layer2,n_channel_layer1,n_channel_layer2])),\
               'fully_connect_layer':tf.Variable(tf.random_normal([(img_size//4)*(img_size//4)*n_channel_layer2,256])),\
               'output':tf.Variable(tf.random_normal([256,action_space_size]))
               }

    Bias = {'conv_layer1': tf.Variable(tf.random_normal([n_channel_layer1])), \
              'conv_layer2': tf.Variable(tf.random_normal([n_channel_layer2])), \
              'fully_connect_layer': tf.Variable(tf.random_normal([256])), \
              'output': tf.Variable(tf.random_normal([action_space_size])),
            }

    #Variables of target neural networks
    Weights_target = {'conv_layer1':tf.Variable(tf.random_normal([filter_size_layer1,filter_size_layer1,frame_dim,n_channel_layer1])),\
               'conv_layer2':tf.Variable(tf.random_normal([filter_size_layer2,filter_size_layer2,n_channel_layer1,n_channel_layer2])),\
               'fully_connect_layer':tf.Variable(tf.random_normal([(img_size//4)*(img_size//4)*n_channel_layer2,256])),\
               'output':tf.Variable(tf.random_normal([256,action_space_size]))
               }

    Bias_target = {'conv_layer1': tf.Variable(tf.random_normal([n_channel_layer1])), \
              'conv_layer2': tf.Variable(tf.random_normal([n_channel_layer2])), \
              'fully_connect_layer': tf.Variable(tf.random_normal([256])), \
              'output': tf.Variable(tf.random_normal([action_space_size])),
            }

    #Process current state
    layer1_convolved_current = convolution2D(s_current,Weights['conv_layer1'])
    layer1_convolved_current = tf.nn.relu(layer1_convolved_current)

    layer2_convolved_current = convolution2D(layer1_convolved_current,Weights['conv_layer2'])
    layer2_convolved_current = tf.nn.relu(layer2_convolved_current)

    flatten_current = tf.reshape(layer2_convolved_current,[-1,(img_size//4)*(img_size//4)*n_channel_layer2])

    full_connect_current = tf.add(tf.matmul(flatten_current,Weights['fully_connect_layer']),Bias['fully_connect_layer'])
    full_connect_current = tf.nn.relu(full_connect_current)

    output_current = tf.add(tf.matmul(full_connect_current,Weights['output']),Bias['output'])

    # Process the next state with the another target network
    layer1_convolved_next = convolution2D(s_next, Weights_target['conv_layer1'])
    layer1_convolved_next = tf.nn.relu(layer1_convolved_next)

    layer2_convolved_next = convolution2D(layer1_convolved_next, Weights_target['conv_layer2'])
    layer2_convolved_next = tf.nn.relu(layer2_convolved_next)

    flatten_next = tf.reshape(layer2_convolved_next, [-1, (img_size//4)*(img_size//4)*n_channel_layer2])

    full_connect_next = tf.add(tf.matmul(flatten_next, Weights_target['fully_connect_layer']),
                               Bias_target['fully_connect_layer'])
    full_connect_next = tf.nn.relu(full_connect_next)

    output_next = tf.add(tf.matmul(full_connect_next, Weights_target['output']), Bias_target['output'])

    #copy the values
    #Copy weights
    global copy_weight_conv1_variable
    copy_weight_conv1_variable = Weights_target['conv_layer1'].assign(Weights['conv_layer1'])
    global copy_weight_conv2_variable
    copy_weight_conv2_variable = Weights_target['conv_layer2'].assign(Weights['conv_layer2'])
    global copy_weight_fully_variable
    copy_weight_fully_variable = Weights_target['fully_connect_layer'].assign(Weights['fully_connect_layer'])
    global copy_weight_output_variable
    copy_weight_output_variable = Weights_target['output'].assign(Weights['output'])
    #Copy bias
    global copy_bias_conv1_variable
    copy_bias_conv1_variable = Bias_target['conv_layer1'].assign(Bias['conv_layer1'])
    global copy_bias_conv2_variable
    copy_bias_conv2_variable = Bias_target['conv_layer2'].assign(Bias['conv_layer2'])
    global copy_bias_fully_variable
    copy_bias_fully_variable = Bias_target['fully_connect_layer'].assign(Bias['fully_connect_layer'])
    global copy_bias_output_variable
    copy_bias_output_variable = Bias_target['output'].assign(Bias['output'])

    return output_current,output_next,r

#Copy target Q-network function
def copy_target_network(session,copy_list):
    for copy_actions in copy_list:
        session.run(copy_actions)

#Action Encoding
def one_hot_encode(action):
    if action not in range(action_space_size):
        raise ValueError('Action could only be 0 or 1')
    else:
        encoded_array = np.ones((action_space_size),dtype=np.int32)
        encoded_array[action] = 0

    return encoded_array.tolist()


#Reward encoding
def reward_transform(initial_reward):
    if initial_reward >=1:
        return 1
    elif initial_reward <=-1:
        return -1
    else:
        return 0

#frame pre-processing function
def Frame_pre_processing(image_data,Pong_Flag=False):
    img = Image.fromarray(image_data, 'RGB').convert('L')
    img = img.resize((img_size,img_size),resample=Image.BILINEAR)
    img_array = np.asarray(img, dtype=np.uint8)
    if Pong_Flag == True:
        img_array.setflags(write=True)
        img_array[img_array>100] = 255

    return img_array

def online_train_neural_network(s_current,r):
    prediction_current,prediction_next,reward_tensor = convolutional_neural_network(s_current,s_next,r)
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
    optimiser = tf.train.RMSPropOptimizer(learning_rate=LR,decay=0.9,momentum=0,centered=True).minimize(loss)
    # optimiser = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

    #Define replay buffer
    replay_buffer = []
    #count the episode that we start to train
    train_start_episode = 0

    #Create the variables that could store the information during training
    Bellman_loss_collection = []
    discounted_reward_collection = []
    total_score_collection = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        copy_target_network(sess, [copy_weight_conv1_variable, copy_weight_conv2_variable,
                                   copy_weight_fully_variable, copy_weight_output_variable,
                                   copy_bias_conv1_variable, copy_bias_conv2_variable,
                                   copy_bias_fully_variable, copy_bias_output_variable])    #Initial copy of the network
        total_step = 0
        for c_episode in range(n_episode):
            # print_flag = False          #For test usage
            if train_start_episode!=0 and (c_episode - train_start_episode)%3 == 0:
                #Copy the network to the target network every 5 steps
                copy_target_network(sess, [copy_weight_conv1_variable, copy_weight_conv2_variable,
                                           copy_weight_fully_variable, copy_weight_output_variable,
                                           copy_bias_conv1_variable, copy_bias_conv2_variable,
                                           copy_bias_fully_variable, copy_bias_output_variable])
                print('Target Q-network updated')
            Frame_buffer = []
            episode_loss = 0
            reward = 0
            run_eps_len = 0
            total_score = 0
            discounted_reward = 0
            env.reset()      #Reset the initialization and start a new episode
            prev_stat_obs = []      #No initial observation
            processed_state = []
            for t in range(episode_len):
                # env.render()             #Display the gaming result
                if t<4:
                    action = env.action_space.sample()   #Random action
                    stat_obs, reward, done, info = env.step(action)
                else:
                    input_state_current = np.array([prev_stat_obs])
                    current_reward = np.asarray([[reward]])
                    action_dict = {s_current:input_state_current,s_next:input_state_current,y_action:np.zeros((1,action_space_size)),r:current_reward}
                    #Using rejection sampling to get a random number
                    uni_sample = random.uniform(0,1)
                    if uni_sample>Epsilon:
                        action = np.asscalar(action_selected.eval(action_dict))   # Exploit the best action
                    else:
                        action = env.action_space.sample()  # Random action
                    # print(prediction_current.eval(action_dict))
                    # print(action)
                    stat_obs, reward, done, info = env.step(action)
                #Process the frames
                grey_obs = Frame_pre_processing(stat_obs,Pong_Flag=False)
                #Append frame
                Frame_buffer.append(grey_obs)
                if len(Frame_buffer)>frame_dim:
                    Frame_buffer.pop(0)
                if len(Frame_buffer) == frame_dim:
                    processed_state = np.stack(Frame_buffer,axis=-1)
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
                    # for item in learn_state_current_list:
                    #     print(np.shape(item))
                    learn_state_current = np.stack(learn_state_current_list,axis=0)
                    learn_state_next = np.stack(learn_state_next_list,axis=0)
                    learn_action = np.stack(learn_action_list,axis=0)
                    learn_reward = np.stack(learn_reward_list,axis=0)
                    learn_dict = {s_current:learn_state_current,s_next:learn_state_next,y_action:learn_action,r:learn_reward}
                    _, currentloss = sess.run([optimiser, loss], feed_dict=learn_dict)
                    episode_loss += currentloss
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
                reward = reward_transform(reward)
                total_score += reward
                discounted_reward += math.pow(Gamma,t)*reward
                run_eps_len = t + 1
                # action_encoded = one_hot_encode(action)
                if done == True:
                    print('Episode ',c_episode+1,' terminated at ',t+1,'steps and the score is:',total_score)
                    break
                elif t == (episode_len-1):
                    print('Episode ',c_episode+1,'does not terminate in all steps. The score is:',total_score)
                if t>=4:
                    replay_buffer.append([prev_stat_obs,action,reward,processed_state])
                prev_stat_obs = processed_state
                if len(replay_buffer)>buffer_len:
                    replay_buffer.pop(0)
                total_step += 1
            print('The loss of ',c_episode+1,'episode is:',episode_loss)
            Bellman_loss_collection.append(episode_loss)
            discounted_reward_collection.append(discounted_reward)
            total_score_collection.append(total_score)
            if total_score>= 30:
                save_model(sess)
            if total_step > 2000000:
                # save_model(sess)
                break
        if total_score_collection[-1]>=10:
            save_model(sess)
        np.save(Save_path+'Bellman_loss',np.asarray(Bellman_loss_collection))
        np.save(Save_path+'raw_score',np.asarray(total_score_collection))
        np.save(Save_path + 'discounted_reward',np.asarray(discounted_reward_collection))

online_train_neural_network(s_current,r)
###################Testing Codes######################
# aa = tf.constant([[8,5],[7,4],[1,2],[4,6]])
# bb = tf.reshape(tf.constant([0,1,1,0]),[4,1])
# ee = tf.contrib.layers.one_hot_encoding(bb,num_classes=2,on_value=0,off_value=1)
# rwo_list = tf.reshape(tf.range(0,limit=4),[4,1])
# indexing_list = tf.concat([rwo_list,bb],axis=1)
# ff = tf.gather_nd(aa,indexing_list)
# with tf.Session() as sess:
#     print(ff.eval())
# for i in range(6):
#     print(one_hot_encode(i))

