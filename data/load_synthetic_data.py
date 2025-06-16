import numpy as np
import torch
from utils.pk_models import schnider_pk
from dask import delayed, compute


#simulating pharmacokinetics/pharmacodynamics while introducing Gaussian noise, returning matrices of concentrations and responses
def simulate_pk_pd(t, k_e=0.1, EC50=1.0, gamma=2.0, ke0=0.3):
    """
    Simulates a basic PK/PD model:
    - PK: 1-compartment decay with elimination constant k_e
    - PD: sigmoid Emax model using Hill equation
    """
    C_p = np.zeros_like(t)  # drug plasma concentration
    C_e = np.zeros_like(t)  # drug effect-site concentration
    R = np.zeros_like(t)  # response (e.g., sedation level)
    dt = t[1] - t[0]

    # Propofol infusion for 1 min
    infusion_duration = 1.0
    infusion_rate = 10.0  # arbitrary units

    for i in range(1, len(t)):
        if t[i] <= infusion_duration:
            input_rate = infusion_rate
        else:
            input_rate = 0.0

        C_p[i] = C_p[i-1] + dt * (input_rate - k_e * C_p[i-1])
        C_e[i] = C_e[i-1] + dt * ke0 * (C_p[i-1] - C_e[i-1])
        R[i] = C_e[i]**gamma / (C_e[i]**gamma + EC50**gamma)

    # Add small Gaussian noise to response
    noise = 0.05 * np.random.randn(len(R))
    R += noise
    R = np.clip(R, 0.0, 1.0)
    C_p = np.clip(C_p, 0, None)  # Clamp to [0, ∞)  
    C_e = np.clip(C_e, 0, None)  # Clamp to [0, ∞)

    return C_p, R, C_e



# More stream-lined addition -- simulating one patient efficiently with Dask
@delayed
def simulate_one_patient(t):
    k_e = np.random.uniform(0.15, 0.25)
    EC50 = np.random.uniform(0.8, 1.2)
    gamma = np.random.uniform(2.5, 3.5)

    C_p, R, C_e = simulate_pk_pd(t, k_e=k_e, EC50=EC50, gamma=gamma)
    return R[0], R[:, None], C_e[:, None]  # y0, true_y, C_e

# dask allows each individual patient to be done in parallel 
def load_synthetic_data_dask(n_patients=32, n_timesteps=100, t_max=10.0):
    t = np.linspace(0, t_max, n_timesteps)

    results = [simulate_one_patient(t) for _ in range(n_patients)]
    results = compute(*results)  # Triggers parallel execution

    y0, true_y, C_all = zip(*results)
    C_all_np = np.clip(np.array(C_all), 0, None)

    C_mean, C_std = C_all_np.mean(), C_all_np.std()
    C_all_np = (C_all_np - C_mean) / C_std

    t_tensor = torch.tensor(t, dtype=torch.float32)
    y0_tensor = torch.tensor(y0, dtype=torch.float32)
    true_y_tensor = torch.tensor(np.array(true_y), dtype=torch.float32)
    C_tensor = torch.tensor(C_all_np, dtype=torch.float32)

    return t_tensor, y0_tensor, true_y_tensor, C_tensor




# If we want to tailor towards propofol
def propofol_effect(Ce, E0=100, Emax=100, EC50=2.5, gamma=3.0):
    """
    Simple sigmoid model of effect-site concentration to sedation score (e.g., BIS).
    """
    return E0 - (Emax * Ce**gamma) / (EC50**gamma + Ce**gamma)


def load_synthetic_data_propofol(n_patients=32, n_timesteps=100, t_max=10.0):
    """
    Returns:
        t_tensor: [T]
        y0_tensor: [B, 1]
        true_y_tensor: [B, T, 1]
        C_tensor: [B, T, 1]
    """
    t = np.linspace(0, t_max, n_timesteps)
    true_y = []
    y0 = []
    C_all = []

    for _ in range(n_patients):
        # Sample patient-specific parameters
        dose = np.random.uniform(50, 200)  # mg
        ke0 = np.random.uniform(0.35, 0.55)
        EC50 = np.random.uniform(2.0, 3.0)
        gamma = np.random.uniform(2.5, 3.5)

        # Run Schnider model
        Cp, Ce = schnider_pk(t, dose=dose, ke0=ke0)

        # Compute effect
        R = propofol_effect(Ce, EC50=EC50, gamma=gamma)

        y0.append([R[0]])
        true_y.append(R[:, None])  # [T, 1]
        C_all.append(Ce[:, None])  # [T, 1]

    C_all_np = np.array(C_all)  # [B, T, 1]
    C_all_np = np.clip(C_all_np, 0, None)

    # Normalize Ce globally
    C_mean = C_all_np.mean()
    C_std = C_all_np.std()
    C_all_np = (C_all_np - C_mean) / C_std

    # Convert to tensors
    t_tensor = torch.tensor(t, dtype=torch.float32)                        # [T]
    y0_tensor = torch.tensor(y0, dtype=torch.float32)                     # [B, 1]
    true_y_tensor = torch.tensor(np.array(true_y), dtype=torch.float32)  # [B, T, 1]
    C_tensor = torch.tensor(C_all_np, dtype=torch.float32)               # [B, T, 1]

    return t_tensor, y0_tensor, true_y_tensor, C_tensor





# Uses tensor to load our synthetic data created with simulate_pk_pd(), returning four matrices with time, y0, y, and concentration
def load_synthetic_data(n_patients=32, n_timesteps=100, t_max=10.0):
    """
    Returns:
        t_tensor: [T]
        y0_tensor: [B, 1]
        true_y_tensor: [B, T, 1]
        C_tensor: [B, T, 1]
    """
    t = np.linspace(0, t_max, n_timesteps)
    dt = t[1] - t[0]
    true_y = []
    y0 = []
    C_all = []

    for _ in range(n_patients):
        # Add variability to each patient's PK/PD params
        k_e = np.random.uniform(0.15, 0.25)
        EC50 = np.random.uniform(0.8, 1.2)
        gamma = np.random.uniform(2.5, 3.5)

        C_p, R, C_e = simulate_pk_pd(t, k_e=k_e, EC50=EC50, gamma=gamma)
        #C = C / C.max()
        y0.append([R[0]])
        true_y.append(R[:, None])  # shape [T, 1]
        C_all.append(C_e[:, None])   # shape [T, 1]

    C_all_np = np.array(C_all)  # shape [B, T, 1]
    C_all_np = np.clip(C_all_np, 0, None)
    # Global normalization
    C_mean = C_all_np.mean()
    C_std = C_all_np.std()

    C_all_np = (C_all_np - C_mean) / C_std

    t_tensor = torch.tensor(t, dtype=torch.float32)
    y0_tensor = torch.tensor(y0, dtype=torch.float32)  # shape [B, 1]
    true_y_tensor = torch.tensor(np.array(true_y), dtype=torch.float32)  # shape [B, T, 1]
    C_tensor = torch.tensor(C_all_np, dtype=torch.float32)    # [B, T, 1]

    if y0_tensor.shape[0] == true_y_tensor.shape[0] == C_tensor.shape[0]:
        print("all equal")
    else:
        print(y0_tensor.shape[0], true_y_tensor.shape[0], C_tensor.shape[0])

    return t_tensor, y0_tensor, true_y_tensor, C_tensor
