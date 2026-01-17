import random
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def set_global_seed(seed: int):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flat_params_from_model(model):
    return parameters_to_vector([p for p in model.parameters() if p.requires_grad]).detach().cpu()

def set_flat_params_to_model(model, flat_vec):
    if isinstance(flat_vec, (list, np.ndarray)):
        flat_vec = torch.from_numpy(np.array(flat_vec))
    vector_to_parameters(flat_vec.to(next(model.parameters()).device), [p for p in model.parameters() if p.requires_grad])

def cosine_sim_np(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)

def pairwise_cosine_stats(list_of_flat_grads):
    n = len(list_of_flat_grads)
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sims.append(cosine_sim_np(list_of_flat_grads[i], list_of_flat_grads[j]))
    if len(sims) == 0:
        return 0.0, 0.0
    return float(np.min(sims)), float(np.mean(sims))