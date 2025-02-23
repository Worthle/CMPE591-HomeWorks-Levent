import subprocess


scripts = {
    "1": "multiperc3.py",
    "2": "imconv.py",
    "3": "imdeconv.py"
}

headers = {
    "1": "Train MLP for Position Estimation",
    "2": "Train CNN for Position Estimation",
    "3": "Train D-CNN for Image Generation"
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
    print("NEURAL NETWORK IS TRAINING")
    subprocess.run(["python",sc_nm],check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    exit(1)