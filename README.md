In order to try all methods please run "train.py" or "test.py" respectively.
Both these codes will ask you to enter a number for different methods.
1- MLP Position Estimation
2- CNN Position Estimation
3- DCNN Image Generation
This way you are be able to train or test the networks with only these two codes.



Brief Explaination for each script files:
- sampledata.py: Runs the simulation for collecting data samples. 800 for train, 200 for validation and 200 for test. Saves the data into "datas" folder. _0.pt files are for training, _1.pt for validation and _2.pt for test. Action, image and position data is gathered using the methods from given script and additional geom data collected using "self.data.geom("obj1").xmat" for including geometry of the object as a feature by altering the homework1.py file.
- multiperc3.py/imconv.py/imdeconv.py: These are the training files. When running, they displays the process in the terminal and after finishing the training, they will save the loss plots in the time_plots folder as well as saving the trained model to the main folder.
- testMLP.py/testCNN.py/test.DCNN.py: These files are for test data evalulations. They will use the model on the test data and displays the total loss. The DCNN one also has a commented section where it will save the images from data and generated ones side by side into test_images folder. Altough generally not succesful as it should, there are really good generated images (like the following) from the DCNN model.

![Image Comparison](/test_images/test_side_by_side_1.png)

General Discussions:

- The codes for training and testing are messy as I have tried many different methods for the betterment of the model. I left it as it is (with unnecessary lines commented) in order to showcase that I have tried those things.
- The following is the error/loss curves:


<img src="/time_plots/MLP_Error_Plot.png" alt="MLP LOSS" width="200" style="display: inline-block;"/>
<img src="/time_plots/CNN_Error_Plot.png" alt="CNN LOSS" width="200" style="display: inline-block;"/>
<img src="/time_plots/DCNN_Error_Plot.png" alt="DCNN LOSS" width="200" style="display: inline-block;"/>
