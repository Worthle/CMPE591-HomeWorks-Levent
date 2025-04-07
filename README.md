**In order to try all methods please run "train.py" or "test.py" respectively.**
Both these codes will ask you to enter a number for different methods.  
1- MLP Position Estimation  
2- CNN Position Estimation  
3- DCNN Image Generation  
4- DQN Training
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
<span style="color:red;">I have switched to a different computer and these images are lost during new push. I will add them again later when i get back to the old computer</span>
<p align="center">
  <img src="/HW2/Reward_Plots/firsttries.png" alt="Normal Epsilon Decay" width="300" style="display: inline-block;"/>
  <img src="/HW2/Reward_Plots/differentrewards.png" alt="Dynamic Epsilon Decay - 100 Episode" width="300" style="display: inline-block;"/>
  <img src="/HW2/Reward_Plots/firstgoodone.png" alt="Dynamic Epsilon Decay - 500 Episode" width="300" style="display: inline-block;"/>
</p>

- The 10000 episode run results are the following. It is obvious that the the learning occurs beacuse the average reward is increased. But in the runs there are still ver low rewards. So the reward window is increased but it doesn't actually converge to a higher value. I don't know why. The first plot is the rewards in the training wher 100 episode took place. Second plot is the test rewards wher I used the trained model in 10 trial runs.  The Plots (Training Rewards - Testing Rewards):

<p align="center">
  <img src="/HW2/Reward_Plots/goodplot.png" alt="Normal Epsilon Decay" width="300" style="display: inline-block;"/>
  <img src="/HW2/Reward_Plots/test.png" alt="Dynamic Epsilon Decay - 100 Episode" width="300" style="display: inline-block;"/>
</p>


**Levent Faruk Soysal**


