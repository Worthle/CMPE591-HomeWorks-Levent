import matplotlib.pyplot as plt
import numpy as np


def mov_avg(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    pad = window_size // 2
    data_padded = np.pad(data, pad_width=pad,mode='edge')
    return np.convolve(data_padded,window,'valid')


rewards = np.load('VPG/rewardsVPG10020_990.npy')
RPS = np.load('VPG/RPSVPG10020_990.npy')

rewards_2 = np.load('SAC/rewardsSAC10020_860.npy')
RPS_2 = np.load('SAC/RPSSAC10020_830.npy')



plt.subplot(2,2,1)
plt.plot(rewards, '-', color='gray', linewidth=0.1)
plt.plot(mov_avg(rewards, 100)[0:-1], 'r', linewidth=1)
plt.title("VPG - Rewards")
plt.subplot(2,2,2)
plt.plot(RPS, '-', color='gray', linewidth=0.1)
plt.plot(mov_avg(RPS, 100)[0:-1], 'r', linewidth=1)
plt.title("VPS - Rewards per Episode Steps")
plt.subplot(2,2,3)
plt.plot(rewards_2, '-', color='gray', linewidth=0.1)
plt.plot(mov_avg(rewards_2, 100)[0:-1], 'r', linewidth=1)
plt.title("SAC - Rewards")
plt.subplot(2,2,4)
plt.plot(RPS_2, '-', color='gray', linewidth=0.1)
plt.plot(mov_avg(RPS_2, 100)[0:-1], 'r', linewidth=1)
plt.title("SAC - Rewards per Episode Steps")
plt.show()