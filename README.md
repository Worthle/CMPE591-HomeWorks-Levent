**In order to try all methods please run "train.py" or "test.py" respectively.**
Both these codes will ask you to enter a number for different methods.  
1- MLP Position Estimation  
2- CNN Position Estimation  
3- DCNN Image Generation  
4- DQN Training  
5- VPG Training  
6- SAC Training  
This way you are be able to train or test the networks with only these two codes.

# HomeWork 1:

**Brief Explaination for each script files:**
- sampledata.py: Runs the simulation for collecting data samples. 800 for train, 200 for validation and 200 for test. Saves the data into "datas" folder. _0.pt files are for training, _1.pt for validation and _2.pt for test. Action, image and position data is gathered using the methods from given script and additional geom data collected using "self.data.geom("obj1").xmat" for including geometry of the object as a feature by altering the homework1.py file.
- multiperc3.py/imconv.py/imdeconv.py: These are the training files. When running, they displays the process in the terminal and after finishing the training, they will save the loss plots in the time_plots folder as well as saving the trained model to the main folder.
- testMLP.py/testCNN.py/test.DCNN.py: These files are for test data evalulations. They will use the model on the test data and displays the total loss. The DCNN one also has a commented section where it will save the images from data and generated ones side by side into test_images folder. Altough generally not succesful as it should, there are really good generated images (like the following) from the DCNN model.

![Image Comparison](/test_images/test_side_by_side_1.png)

**General Discussions:**

- The codes for training and testing are messy as I have tried many different methods for the betterment of the model. I left it as it is (with unnecessary lines commented) in order to showcase that I have tried many things. Same thing goes for the first/second/third set folders in datas. I used them for different trails and didn't wanted to get rid of them.
- The following is the error/loss curves:
<p align="center">
  <img src="/time_plots/MLP_Error_Plot.png" alt="MLP LOSS" width="300" style="display: inline-block;"/>
  <img src="/time_plots/CNN_Error_Plot.png" alt="CNN LOSS" width="300" style="display: inline-block;"/>
  <img src="/time_plots/DCNN_Error_Plot.png" alt="DCNN LOSS" width="300" style="display: inline-block;"/>
</p>

  It can be seen that the curves converging to a minimum point. One this I couldn't understand is that why the validation loss is less than training loss for Position Estimation Models. Also, for Positon estimation, both models work really good with Loss is around <0.005. But for Image generation, not that great. Generally, all three models are sufficient enough to be described as a "trained model".
- Another issue I have faces is the size. I couldn't make enough trails for DCNN primarily due to the size of the data and model. I couldn't even work on the total 800 training data due to this. But I could get some sensible outcome from each model so I believe it is sufficient.
- Every single hyper parameter like kernel size or stride of convolution layers, hidden ayer size, activation functions etc. are selected after many trials. The final ones are, as far as I can try, the best ones.

**Levent Faruk Soysal**

# HomeWork 2:

**Brief Explaination**
- For Deep Q Learning assignment there are only two codes. (aside from environment.py and homework2.py)
- You can evaluate and run the codes by using the train.py and test.py for this assignment too. (**And for all the assignments in the future.**)
- trainDQN.py: This is the folder where Q Learning and Neural Network Training has implemented. Replay Buffer is defined using OOP as well as Neural Network. Network has one hidden laer with nodes. For optimizer, RMSProp and Huber Loss (SmoothL1) is used as stated in lecture notes that these methods yields better results in RL. Although I tried other methods too, they were not that satisfactory. The following is the hypermaters I used in my best resulted training:
  
learning_rate = 0.0001  
num_episodes = 10000  
update_frequency = 10  
target_update_frequency = 200  
epsilon = 1.0  
epsilon_decay = 0.9995  
epsilon_decay_high_reward = 0.8  
epsilon_increase = 1.001  
epsilon_min = 0.05  
epsilon_max = 1.0  
batch_size=64  
gamma=0.9  
buffer_size=50000  
reward_threshold = 10.0  
high_reward_threshold = 50.0  

In here, buffer size is selected as 50000. For epsilon, there are many parameters as I have tried to have somewhat of a dynamic epsilon (reward thresholds and additional epsilon parameters), but later decided that to have usual decay.
Before training the neural network, I fill the buffer with 50000 samples. After training, I save the model and plot the Reward, Reward per #Episode and epsilon.
- TestDQN.py: In testing, Instead of "offscreen", "gui" is used for environment so we can see what's happening. Epsilon is kept at 0.0 meaning no exploration. So the model only take action based on the training.

**Discussions and Results**
- Since this is a computation heavy task, I could not have time to make many different trials with different parameters. For testing the code and fixing bugs etc. there was not an option for me to have reliable trails. So, only before the last training session, I could only use like 100-200 episodes for testing due to this issue. My final run took about 30 hours o train.
- Second note is that I changed the maximum step time (self._max_timesteps = 100) in the homework2.py file to 100. This resulted better in smaller episode runs, and in general. 
- The following are the plots in my trials. The first one is my first try where I tweak the parameters and this is the best I could get. Other ones when I implemented dynamic epsilon. Meaning that for two threshold values for reward, epsilon changes differently. If the reward is less than 10, epsilon is increase, therefore more exploration. If it is between 10-50, epsilon is decrased by a small value. If reward is more than 50, then epsilon is decreased by a larger value. So, this is all for tweaking the exploration and exploitation. The results are better than conventional epsilon decaying but I don't want to take the risk for higher episodes, because of computation time. What if the epsilon stays always at max? I couldn't take that chance so for 10000 episode run, I didn't used dynaimc epsilon.  
  The Plots (Normal Epsilon Decay / Dynamic Epsilon Decay - 100 Episode / Dynamic Epsilon Decay - 500 Episode):
  !!!!!!!!!!!!!!!!!!!I have switched to a different computer and these images are lost during new push. I will add them again later when i get back to the old computer!!!!!!!!!!!!!!!
<p align="center">
  <img src="/test_images/firsttries.png" alt="Normal Epsilon Decay" width="300" style="display: inline-block;"/>
  <img src="/test_images/differentrewards.png" alt="Dynamic Epsilon Decay - 100 Episode" width="300" style="display: inline-block;"/>
  <img src="/test_images/firstgoodone.png" alt="Dynamic Epsilon Decay - 500 Episode" width="300" style="display: inline-block;"/>
</p>

- The 10000 episode run results are the following. It is obvious that the the learning occurs beacuse the average reward is increased. But in the runs there are still ver low rewards. So the reward window is increased but it doesn't actually converge to a higher value. I don't know why. The first plot is the rewards in the training wher 100 episode took place. Second plot is the test rewards wher I used the trained model in 10 trial runs.  The Plots (Training Rewards - Testing Rewards):

<p align="center">
  <img src="/test_images/goodplot.png" alt="Normal Epsilon Decay" width="300" style="display: inline-block;"/>
  <img src="/test_images/test.png" alt="Dynamic Epsilon Decay - 100 Episode" width="300" style="display: inline-block;"/>
</p>




# HomeWork 3:

For this homework, I have tried everything and couldn't get a decent result. Therefore, I will only discuss the things I have tried.

**Brief Explaination of the Codes**

- For Vanilla Policy Gradient, the VPG class is fot the policy gradient network and agent class is the primary class. Decide action by sampling action probabilities. Then, in update model it calculates the loss using the rewards and back propogate. 
- For Soft Action Critic, Actor Network is similar to the VPG, and Two Critic Network are implemented for the Q-value estimation. Alpha update is immplemented for opimizing the entropy for ensuring the policy maintains enough exploration. Then targer network is being updated (soft update). 
For both network, save, plot and moving average functions are added for visualization.

**Discussion**
- Final trained model which yielded the best results are the ones with model10020_n.pt where n is the episode number. Latest episode doesn't have the highest rewards but the plotting.py and test.py uses that so VPGmodel10020_1000.pt and SACactormodel10020_1000.pt, SACcritic1model10020_1000.pt, SACcritic2model10020_1000.pt. There are many other trained models with higher or lower episodes, feel free to check them out but nothing will work :D
- Main Problem is that, no matter what I do, the model is moving to a point regardless of positions of goal or object. No matter the length of training or the hyperparameters I couldn't fix this problem. Also, I have tried to look for a problem in my implementations, but couldn't debug the issue.
- **Parameter Tests:** I have tried different gammas, learning rates, hidden layers sizes, number of hidden layers and many other parameters. None of it worked properly. I also tried to fix the position of the goal and object for every iterations, didn't quite helped.
- **Reward Structure:** My final attempt is to change the reward structure. I have tried smaller weights for discounts provided and added three more things. One is that since generally the robot stucks to the sides of the plane, I added a discount if the end effector position is not changed too much there will be a discount. Second thing is, I incorporated the end effector position to the goal as well as end effector to the object. Since the grid is 0.3x0.3, I by using the approximate maximum difference (hypotenuse 0.3*sqrt(2) = 0.234) between end effector and object and normalized the distance. for directional reward, instead of the value given 0.5 I used 1 - norm_ee_to_obj. And I added directional reward for the end effector two with the weight of norm_ee_to_obj. And these two are the total directional reward. This way, it add rewards for both object and end effector direction to goal while prioritizing the object direction. Third thing I added is when the end effector is close to the object, it gets a large positive rewards and if not get a small discount. Also, when running the code, it will always say high reward. I put that to see positive rewards more easily but since I changed the reward structure all the rewards are positive. I left it there in case you want to test with the given reward structure.

**Results:** After the reward changes, I could finally get a increase in reward through episodes but only for Soft Actor Critic. But the tests are still not sufficient. I have saved almost all the results in the VPG and SAC folders, but for testing parameters vary, so not all trained model might work directly and need some configuration in the plotting.py script. Also, the moving average does strange thing at the start and end of the plots but I couldn't fix that too and It is not a big problem because it can be understood easily. The following are the 1000 episode runs for the final version where the additional rewards are added. For VPG, I used a 4 layer network, which sizes are [16 32 32 16]. For SAC, Actor network layer size is 512 and critic networks are 256. Reward is incresing for both models but end effector is moving to goal without the object. I tried to change the weights in my reward structure but couldn't solve this. It is either goes to a random point like always, or moves to goal. In the test simulations for the final method, there is hope but it is not quite there yet. Maybe with more training it will finally learn but there was no more time left. (I will add it later if it works with late submission disclaimer.)

# HomeWork 3 Extended:

Now that working in Gym environment, I could manage to get increase in rewards for Vanilla Policy Gradient, but SAC is still bad. 
<p align="center">
  <img src="/VPG/training_progress_gym_vpg_276000.png" alt="VPG" width="300" style="display: inline-block;"/>
  <img src="/SAC/training_progress_gym_sac_20000.png" alt="SAC" width="300" style="display: inline-block;"/>
</p>

# HomeWork 4:

<p align="center">
  <img src="/time_plots/cnmperror.png" alt="Normal Epsilon Decay" width="300" style="display: inline-block;"/>
  <img src="/time_plots/cnmppred.png" alt="Dynamic Epsilon Decay - 100 Episode" width="300" style="display: inline-block;"/>
  <img src="/time_plots/cnmpbar.png" alt="Dynamic Epsilon Decay - 500 Episode" width="300" style="display: inline-block;"/>
</p>















**Levent Faruk Soysal**


