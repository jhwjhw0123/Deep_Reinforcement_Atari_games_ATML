#Linear
import numpy as np
import matplotlib.pyplot as plt
LR = 0.5
Save_path = '../../result/A.3/Nonlinear/'
bellmanLoss = np.load(Save_path+'Bellmanloss linear learning rate = '+str(LR)+'.npy')
step_len = np.load(Save_path + 'Performance_length with LR=' + str(LR)+'.npy')
discounted_reward = np.load(Save_path + 'discounted_reward with LR=' + str(LR)+'.npy')
plt.subplot(1,3,1)
plt.plot(bellmanLoss)
plt.xlabel('Bellman Loss')
plt.subplot(1,3,2)
plt.plot(step_len)
plt.xlabel('Step Length')
plt.subplot(1,3,3)
plt.plot(discounted_reward)
plt.xlabel('Discounted Reward')
plt.show()