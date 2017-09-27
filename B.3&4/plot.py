#Linear
import numpy as np
import matplotlib.pyplot as plt
Save_path = '../../result/B.3&4/Pong/'
bellman_loss = np.load(Save_path+'Bellman_loss.npy')
raw_score = np.load(Save_path+'raw_score.npy')
Dis_score = np.load(Save_path+'discounted_reward.npy')
index_50K = np.array(range(0,Dis_score.shape[0],15))
bellman_loss_log = np.log(bellman_loss)
raw_score_50K = raw_score[index_50K]
Dis_score_50K = Dis_score[index_50K]
# plt.subplot(1,3,1)
# plt.plot(bellman_loss_log)
# plt.title('Bellman log Loss')
# plt.subplot(1,3,2)
# plt.plot(raw_score)
# plt.title('Raw Score')
# plt.subplot(1,3,3)
plt.title('Selected Agent Performance Performance')
plt.subplot(1,2,1)
plt.plot(raw_score_50K)
plt.xlabel('Episode')
plt.ylabel('raw_score')
plt.subplot(1,2,2)
plt.plot(Dis_score_50K)
plt.xlabel('Episode')
plt.ylabel('Discounted_score')
plt.show()