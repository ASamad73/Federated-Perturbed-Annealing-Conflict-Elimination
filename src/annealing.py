import numpy as np
import torch
from copy import deepcopy
from torch.nn.utils import parameters_to_vector
from utils import set_flat_params_to_model

def sample_k_seeds(K, base_seed=None):
    rng = np.random.RandomState(base_seed)
    return [int(rng.randint(0, 2**31 - 1)) for _ in range(K)]

def make_uniform_delta_from_seed(seed, prototype_vector, rho, device='cpu', scale_by_norm=False):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) & 0xffffffff)
    flat = prototype_vector.to(device)
    uni = (torch.rand(flat.shape, generator=gen, device=device) * 2.0 - 1.0) * rho
    if scale_by_norm:
        model_norm = torch.norm(flat) + 1e-12
        delta = uni * model_norm
    else:
        delta = uni
    return delta

def client_compute_scores_for_fedgga(model, loss_fn, data_loader, seeds, rho, device='cpu', 
                                     scale_by_norm=False):
    
    local_model = deepcopy(model).to(device)
    local_model.train()

    it = iter(data_loader)
    xs, ys = next(it)
    xs, ys = xs.to(device), ys.to(device)

    param_list = [p for p in local_model.parameters() if p.requires_grad]
    
    if len(param_list) == 0:
        return [0.0] * len(seeds), np.zeros(1), 0.0, [0.0] * len(seeds)
        
    numels = [p.numel() for p in param_list]
    flat_theta = parameters_to_vector(param_list).detach().to(device)

    local_model.zero_grad()
    out = local_model(xs)
    loss_ref_tensor = loss_fn(out, ys)
    loss_ref = float(loss_ref_tensor.detach().cpu().item())
    loss_ref_tensor.backward()

    ref_grad_parts = []
    for p in param_list:
        g = p.grad
        if g is None:
            ref_grad_parts.append(torch.zeros(p.numel(), device=device))
        else:
            ref_grad_parts.append(g.detach().view(-1))
            
    ref_grad_t = torch.cat(ref_grad_parts)            
    ref_grad_numpy = ref_grad_t.detach().cpu().numpy()

    scores = []
    losses_k = []

    def apply_delta_inplace(delta_flat):
        offset = 0
        for p, n in zip(param_list, numels):
            seg = delta_flat[offset: offset + n].view_as(p.data)
            p.data.add_(seg)
            offset += n

    def revert_delta_inplace(delta_flat):
        offset = 0
        for p, n in zip(param_list, numels):
            seg = delta_flat[offset: offset + n].view_as(p.data)
            p.data.sub_(seg)
            offset += n

    for seed in seeds:
        delta = make_uniform_delta_from_seed(seed, flat_theta, rho, device=device, scale_by_norm=scale_by_norm)
        apply_delta_inplace(delta)

        local_model.zero_grad()
        out_k = local_model(xs)
        loss_k_tensor = loss_fn(out_k, ys)
        loss_k = float(loss_k_tensor.detach().cpu().item())
        loss_k_tensor.backward()

        gk_parts = []
        for p in param_list:
            g = p.grad
            if g is None:
                gk_parts.append(torch.zeros(p.numel(), device=device))
            else:
                gk_parts.append(g.detach().view(-1))
        gk_t = torch.cat(gk_parts)

        denom = (torch.norm(gk_t) * torch.norm(ref_grad_t) + 1e-12)
        sim_t = float(torch.dot(gk_t, ref_grad_t).item() / denom.item())

        scores.append(sim_t)
        losses_k.append(loss_k)

        revert_delta_inplace(delta)

    set_flat_params_to_model(local_model, flat_theta)

    return scores, ref_grad_numpy, loss_ref, losses_k
