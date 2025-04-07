import matplotlib.pyplot as plt
import numpy as np


def mov_avg(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data,window,'same')


rewards = np.load('VPG/rewardsVPG10010_2600.npy')
RPS = np.load('VPG/RPSVPG10010_2600.npy')


plt.subplot(1,2,1)
plt.plot(rewards, '-', color='gray', linewidth=0.1)
plt.plot(mov_avg(rewards, 100)[100:-100], 'r', linewidth=1)
plt.title("Rewards")
plt.subplot(1,2,2)
plt.plot(RPS, '-', color='gray', linewidth=0.1)
plt.plot(mov_avg(RPS, 100)[100:-100], 'r', linewidth=1)
plt.title("Rewards per Episode Steps")
plt.show()