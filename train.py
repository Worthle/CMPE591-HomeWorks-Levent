import subprocess


scripts = {
    "1": "multiperc3.py",
    "2": "imconv.py",
    "3": "imdeconv.py",
    "4": "trainDQN.py",
    "5": "trainVPG.py",
    "6": "trainSAC.py"
}

headers = {
    "1": "Train MLP for Position Estimation - HW1",
    "2": "Train CNN for Position Estimation - HW1",
    "3": "Train D-CNN for Image Generation - HW1",
    "4": "Train DQN - HW2",
    "5": "Train VPG - HW3",
    "6": "Train SAC - HW3"
}
print("Select A Training Method: ")
for key,value in headers.items():
    print(f"{key}:{value}")

choice = input("Enter the number of the script to run: ")
sc_nm = scripts.get(choice)

if not sc_nm:
    print("Invalid Number. Exiting.")
    exit(1)
try:
    if choice == "4":
        print("DQN TRAINING INITIALIZING")
    elif choice == "5":
        print("VPG TRAINING INITIALIZING")
    elif choice == "6":
        print("SAC TRAINING INITIALIZING")
    else:
        print("NEURAL NETWORK IS TRAINING")
    subprocess.run(["python",sc_nm],check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    exit(1)