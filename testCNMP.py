import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
# === 1. Load Data ===
X_train = np.load("training_X.npy")
Y_train = np.load("training_Y.npy")
X_val = np.load("validation_X.npy")
Y_val = np.load("validation_Y.npy")

# === 2. Model Definition ===
class CNMP(nn.Module):
    def __init__(self, d_x, d_y, hidden_size=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_x + d_y, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + d_x, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2 * d_y)
        )

    def forward(self, context, query):
        r = self.encoder(context).mean(dim=0, keepdim=True)
        r = r.expand(query.size(0), -1)
        output = self.decoder(torch.cat([r, query], dim=1))
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.nn.functional.softplus(log_std) + 1e-6
        return mean, std

# === 3. Training Utilities ===
def get_sample(X, Y, obs_max=5):
    d = np.random.randint(0, X.shape[0])
    idx = np.random.permutation(X.shape[1])
    n = np.random.randint(1, obs_max + 1)
    ctx = np.hstack([X[d, idx[:n]], Y[d, idx[:n]]])
    query = X[d, idx[n]].reshape(1, -1)
    target = Y[d, idx[n]].reshape(1, -1)
    return torch.tensor(ctx, dtype=torch.float64), torch.tensor(query, dtype=torch.float64), torch.tensor(target, dtype=torch.float64)

# === 4. Main ===
if __name__ == "__main__":
  
    torch.set_default_dtype(torch.float64)
    d_x, d_y = X_train.shape[-1], Y_train.shape[-1]
    model = CNMP(d_x, d_y)
    model.load_state_dict(torch.load("cnmp_model.pth"))
    n_tests = 100
    end_errors, obj_errors = [], []

    model.eval()
    for _ in range(n_tests):
        d = np.random.randint(0, X_val.shape[0])
        idx = np.random.permutation(X_val.shape[1])
        n_ctx = np.random.randint(1, 6)
        n_qry = np.random.randint(1, 21)

        ctx = torch.tensor(np.hstack([X_val[d, idx[:n_ctx]], Y_val[d, idx[:n_ctx]]]), dtype=torch.float64)
        qry = torch.tensor(X_val[d, idx[n_ctx:n_ctx+n_qry]], dtype=torch.float64)
        tgt = torch.tensor(Y_val[d, idx[n_ctx:n_ctx+n_qry]], dtype=torch.float64)

        mean, _ = model(ctx, qry)
        err_end = torch.nn.functional.mse_loss(mean[:, :2], tgt[:, :2]).item()
        err_obj = torch.nn.functional.mse_loss(mean[:, 2:], tgt[:, 2:]).item()

        end_errors.append(err_end)
        obj_errors.append(err_obj)

    end_errors = np.array(end_errors)
    obj_errors = np.array(obj_errors)

    plt.figure()
    plt.bar([0, 1], [end_errors.mean(), obj_errors.mean()], yerr=[end_errors.std(), obj_errors.std()], capsize=5)
    plt.xticks([0, 1], ['End-Effector', 'Object'])
    plt.ylabel('MSE')
    plt.title('Prediction Errors (Mean ± Std)')
    plt.show()

    model.eval()
    # Pick a random validation trajectory
    d = random.randint(0, X_val.shape[0] - 1)
    idx = np.random.permutation(X_val.shape[1])
    n_ctx = 5  # Number of context points (fixed for visualization)

    # Prepare context points
    ctx_idx = idx[:n_ctx]
    ctx = torch.tensor(np.hstack([X_val[d, ctx_idx], Y_val[d, ctx_idx]]), dtype=torch.float64)

    # Query the entire trajectory
    qry = torch.tensor(X_val[d], dtype=torch.float64)
    tgt = torch.tensor(Y_val[d], dtype=torch.float64)
    mean, _ = model(ctx, qry)
    mean = mean.detach().numpy()

    # Plotting
    plt.figure()
    # Plot ground truth end-effector (e_y, e_z) trajectory
    plt.plot(tgt[:, 0], tgt[:, 1], 'g-', label='Ground Truth')
    # Plot predicted end-effector trajectory
    plt.plot(mean[:, 0], mean[:, 1], 'k-', label='Predicted')
    # Plot context points (e_y, e_z)
    plt.scatter(mean[ctx_idx, 0], mean[ctx_idx, 1], c='black', marker='*', s=100, label='Context')
    plt.legend()
    plt.title('End-Effector Trajectory Prediction')
    plt.xlabel('e_y')
    plt.ylabel('e_z')
    plt.show()