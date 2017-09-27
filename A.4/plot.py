#Linear
import numpy as np
import matplotlib.pyplot as plt
Save_path = '../../result/A.4/'
# performance = np.load(Save_path+'episode_performance.npy')
performance = np.load(Save_path+'discounted_score_performance.npy')
performance = performance
plt.plot(performance)
plt.xlabel('Episode')
plt.ylabel('Average Length')
plt.show()