import torch
import torch.nn as nn
from torchdiffeq import odeint
from models.ode_func import ODEFunc
from models.ODE_wrapper import ODEWrapper
from models.semiODE import SemiMechanisticODE
from data.load_synthetic_data import load_synthetic_data, load_synthetic_data_propofol, load_synthetic_data_dask
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
# from dask.distributed import Client

# if __name__ == "__main__":
#     client = Client()  # Launches local dashboard at http://localhost:8787
# # client = Client()  

# Hyperparameters
epochs = 100
lr = 1e-3

t, X, Y, C = load_synthetic_data_dask()
# Load synthetic dataset with generic drug 
# t, X, Y, C = load_synthetic_data()
# Load synthetic dataset for specifically propool 
# t, X, Y, C = load_synthetic_data_propofol(n_patients=32)

# Initial state x0 shape [B, dim]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x0 = X
x0 = x0.to(device)
t = t.to(device)
Y = Y.to(device)


# Neural ODE setup
ode_func = ODEFunc().to(device)
# ode_func = SemiMechanisticODE().to(device) 
optimizer = torch.optim.Adam(ode_func.parameters(), lr=lr)
criterion = nn.MSELoss()


#Train/test split (80:20 ratio)
B = Y.shape[0]
train_size = int(0.8 * B)
test_size = B - train_size

train_X = X[:train_size].to(device)
train_Y = Y[:train_size].to(device)

test_X = X[train_size:].to(device)
test_Y = Y[train_size:].to(device)
test_C = C[train_size:].to(device)

print(f"X shape: {X.shape}, Y shape: {Y.shape}, C shape: {C.shape}")
print(f"train_X: {train_X.shape}, test_X: {test_X.shape}")
print(f"train_Y: {train_Y.shape}, test_Y: {test_Y.shape}")
print(f"train_C: {C[:train_size].shape}, test_C: {test_C.shape}")

t_vector = t  # shape [T]
C_train = C[:train_size].to(device)  # shape [B, T, 1]

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # Extract concentrations for training patients: shape [B, T, 1]
    C_train = C[:train_size].to(device) 

    wrapped_ode = ODEWrapper(ode_func, C_train, t_vector)

    if train_X.ndim == 1:
        train_X = train_X.unsqueeze(-1)

    pred_y = odeint(wrapped_ode, train_X, t)
    pred_y = pred_y.permute(1, 0, 2)  # reshape to [batch, T, dim]

    loss = criterion(pred_y, train_Y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Evaluation on test population
with torch.no_grad():
    C_test = C[train_size:].to(device)
    T = t.shape[0]
    t_vector = t.to(device)
    t_max = t_vector[-1].item()

    wrapped_ode_test = ODEWrapper(ode_func, C_test, t)

    if test_X.ndim == 1:
        test_X = test_X.unsqueeze(-1)    

    test_pred_y = odeint(wrapped_ode_test, test_X, t.to(device))
    test_pred_y = test_pred_y.permute(1, 0, 2)  # shape: [batch, T, dim]

# R² for each test patient
r2_scores = []
for i in range(test_size):
    true = test_Y[i, :, 0].cpu().numpy()
    pred = test_pred_y[i, :, 0].cpu().numpy()
    r2_scores.append(r2_score(true, pred))

print(f"\nAverage R² on Test Set: {sum(r2_scores)/len(r2_scores):.4f}")

# Plot a few test patients
for i in range(min(3, test_size)):
    plt.figure(figsize=(8, 4))
    plt.plot(t.cpu().numpy(), test_Y[i, :, 0].cpu().numpy(), label='True PD', color='tab:green')
    plt.plot(t.cpu().numpy(), test_pred_y[i, :, 0].cpu().numpy(), label='Predicted PD', linestyle='--', color='tab:orange')
    plt.plot(t.cpu().numpy(), test_C[i, :, 0].cpu().numpy(), label='Drug Concentration', color='tab:blue', alpha=0.5)
    plt.title(f'Test Patient {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.legend()
    plt.tight_layout()
    plt.show()

