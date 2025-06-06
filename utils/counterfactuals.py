import numpy as np
import torch
from copy import deepcopy

NON_ACTIONABLES = [
    'Gender_Female', 'Gender_Male',
    'MaritalStatus_Divorced', 'MaritalStatus_Single', 'MaritalStatus_Married',
    'CityTier'
]

def generate_counterfactual(instance, model, step_size=0.01, max_iter=1000, threshold=0.5, feature_names=None):
    x_cf = deepcopy(instance)
    for _ in range(max_iter):
        x_tensor = torch.tensor(x_cf, dtype=torch.float32)
        pred = model(x_tensor).item()
        if pred < threshold:
            return x_cf
        perturb = np.random.uniform(-step_size, step_size, size=x_cf.shape)
        if feature_names is not None:
            for j, name in enumerate(feature_names):
                if name in NON_ACTIONABLES:
                    perturb[j] = 0
        x_cf = np.clip(x_cf + perturb, 0, 1)
    return None