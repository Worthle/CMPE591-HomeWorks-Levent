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
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    losses, val_mses = [], []
    for step in range(100000):
        ctx, q, tgt = get_sample(X_train, Y_train)
        optimizer.zero_grad()
        mean, std = model(ctx, q)
        dist = torch.distributions.Normal(mean, std)
        loss = -dist.log_prob(tgt).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 10000 == 0:
            model.eval()
            val_mse = 0
            for i in range(X_val.shape[0]):
                ctx = torch.tensor(np.hstack([X_val[i, :1], Y_val[i, :1]]), dtype=torch.float64)
                q = torch.tensor(X_val[i], dtype=torch.float64)
                tgt = torch.tensor(Y_val[i], dtype=torch.float64)
                mean, _ = model(ctx, q)
                val_mse += torch.nn.functional.mse_loss(mean, tgt).item()
            val_mse /= X_val.shape[0]
            val_mses.append(val_mse)
            print(f"Step {step}, Loss {loss.item():.8f}, Val MSE {val_mse:.8f}")
            model.train()
    torch.save(model.state_dict(), "cnmp_model.pth")
    # Plot
    plt.figure()
    plt.plot(losses, label='Train Loss')
    plt.plot(np.arange(0, len(val_mses)*10000, 10000), val_mses, label='Val MSE')
    plt.legend()
    plt.show()

    