import subprocess


scripts = {
    "1": "testMLP.py",
    "2": "testCNN.py",
    "3": "testDCNN.py",
    "4": "testDQN.py"
}

headers = {
    "1": "Test MLP for Position Estimation - HW1",
    "2": "Test CNN for Position Estimation - HW1",
    "3": "Test D-CNN for Image Generation - HW1 (Example Images at the folder test_images)",
    "4": "Test DQN - HW2"
}
print("Select A Method for Testing the Trained Network: ")
for key,value in headers.items():
    print(f"{key}:{value}")

choice = input("Enter the number of the script to run: ")
sc_nm = scripts.get(choice)

if not sc_nm:
    print("Invalid Number. Exiting.")
    exit(1)
try:
    if choice == "4":
        print("DQN TESTING INITIALIZING")
    else:
        print("LOSS FOR TEST DATA IS CALCULATING")
    
    subprocess.run(["python",sc_nm],check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    exit(1)
