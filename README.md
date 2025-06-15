# Neural ODE Anesthesia Drug Simulator

This project implements a Neural Ordinary Differential Equation (Neural ODE) model to simulate the pharmacokinetics (PK) and pharmacodynamics (PD) of anesthesia drugs. It includes synthetic data generation, effect-site modeling (via ke0), and performance evaluation.

## Features

- Synthetic PK/PD data generation
- Neural ODE modeling using PyTorch
- Effect-site compartment with tunable `ke0`
- Batch training with Gaussian noise
- R² performance metrics
- Visualization of predicted vs true PD responses (see images folder)

## Model Performance

The model achieved an average R² of **0.856** on synthetic test data using `ke0 = 0.1` and added Gaussian noise, demonstrating strong predictive ability for drug response dynamics.

## Project Structure

main.py -> training + evaluation
data_generator.py -> synthetic PK/PD data generation (w/ Gaussian noise)
ode_func.py -> Neural ODE function definitions
utils.py -> pk_models.py has specific example anesthetic drug models (for future implementation)
requirements.txt -> req Python packages
LICENSE -> MIT license
README.md -> project overview + instructions

## Installation

```bash
git clone https://github.com/mjkovoor/AnesthesiaNeuralODE.git
cd AnesthesiaNeuralODE
pip install -r requirements.txt