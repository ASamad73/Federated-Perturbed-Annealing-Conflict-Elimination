import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.cuda.amp import GradScaler, autocast
from annealing import (sample_k_seeds, make_uniform_delta_from_seed,
                        client_compute_scores_for_fedgga)
from utils import pairwise_cosine_stats, get_head_param_list_and_names
from strategies import AgreementWeightedFedAvg, PruningFedAvg
from utils import fedavg_from_state_dicts

@dataclass
class GradientDict:
    """Encapsulates gradients as a dictionary of tensors."""
    gradients: Dict[str, torch.Tensor]
    client_id: str
    
    def sign(self) -> Dict[str, torch.Tensor]:
        return {name: torch.sign(grad) for name, grad in self.gradients.items()}
    
    def to_device(self, device):
        self.gradients = {name: grad.to(device) for name, grad in self.gradients.items()}
        return self
    
    @staticmethod
    def from_model(model: nn.Module, client_id: str) -> 'GradientDict':
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
        return GradientDict(gradients=gradients, client_id=client_id)
    
    def apply_to_model(self, model: nn.Module, learning_rate: float):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.gradients:
                    param.data -= learning_rate * self.gradients[name]

class FedClient:
    def __init__(self, name, train_loader, test_loader, device):
        self.name = name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def score_seeds(self, model, loss_fn, seeds, rho, scale_by_norm):
        return client_compute_scores_for_fedgga(model, loss_fn, self.train_loader, seeds, rho, device=self.device, scale_by_norm=scale_by_norm)

    def local_update(self, global_model, local_epochs=1, lr=0.01, max_steps=None, use_amp=False):
        model = deepcopy(global_model).to(self.device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            return deepcopy(model.state_dict())
            
        opt = torch.optim.Adam(trainable_params, lr=lr)
            
        loss_fn = nn.CrossEntropyLoss()
        model.train()

        scaler = GradScaler() if (use_amp and self.device.startswith('cuda')) else None

        step = 0
        for _ in range(local_epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to( self.device), yb.to( self.device)
                opt.zero_grad()
                if scaler is not None:
                    with autocast():
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
                step += 1
                if (max_steps is not None) and (step >= max_steps):
                    break
            if (max_steps is not None) and (step >= max_steps):
                break

        return deepcopy(model.state_dict())
    
    def compute_avg_gradient(self, global_model, local_epochs=1, max_batches=None, device=None):
        device = device or self.device
        model = deepcopy(global_model).to(device)
        model.train()

        loss_fn = nn.CrossEntropyLoss()

        accumulated = None
        batch_count = 0
        step = 0

        for ep in range(local_epochs):
            for xb, yb in self.train_loader:
                if (max_batches is not None) and (step >= max_batches):
                    break
                xb, yb = xb.to(device), yb.to(device)
                model.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()

                batch_grads = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        batch_grads[name] = param.grad.detach().cpu().clone()

                if accumulated is None:
                    accumulated = {k: v.clone() for k, v in batch_grads.items()}
                else:
                    for k, v in batch_grads.items():
                        if k in accumulated:
                            accumulated[k] += v
                        else:
                            accumulated[k] = v.clone()

                batch_count += 1
                step += 1
            if (max_batches is not None) and (step >= max_batches):
                break

        if accumulated is None:
            return {}

        for k in accumulated:
            accumulated[k] = accumulated[k] / float(batch_count)

        return accumulated  


    def eval_on_test(self, model):
        model = deepcopy(model).to(self.device)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x,y in self.test_loader:
                x,y = x.to(self.device), y.to(self.device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

class FedGGAServer:
    def __init__(self, server_model, clients, device, config, test_loader):
        self.model = deepcopy(server_model).to(device)
        self.clients = clients
        self.device = device
        self.config = config.copy()
        self.log = []
        self.W_history = []
        self.test_loader = test_loader


    def run(self):
        cfg = self.config
        loss_fn = nn.CrossEntropyLoss()
        extra_grad_evals = 0
        for rnd in range(cfg['rounds']):
            client_state_dicts = []
            client_flat_updates = []
            applied = False
            
            t0 = time.time()
            
            if cfg['R_start'] <= rnd <= cfg['R_end'] and cfg['enable_gga']:
                seeds = sample_k_seeds(cfg['K'], base_seed=cfg.get('base_seed', 1234) + rnd)
                
                client_scores = []  
                client_ref_grads = []
                client_ref_losses = []
                client_losses_k = [] 

                for c in self.clients:
                    scores, ref_grad, ref_loss, losses_k = c.score_seeds(self.model, loss_fn, seeds, cfg['rho'], 
                                                             scale_by_norm=cfg.get('scale_by_norm', False))
                    client_scores.append(scores)
                    client_ref_grads.append(ref_grad)
                    client_ref_losses.append(ref_loss)
                    client_losses_k.append(losses_k)
                    extra_grad_evals += (1 + len(seeds)) * 1  

                sim_ref_min, sim_ref_mean = pairwise_cosine_stats(client_ref_grads)
                LB = float(np.mean(client_ref_losses)) 

                arr_scores = np.stack(client_scores, axis=0)  
                avg_scores = np.mean(arr_scores, axis=0)     

                arr_losses_k = np.stack(client_losses_k, axis=0)  # (n_clients, K)
                avg_losses_k = np.mean(arr_losses_k, axis=0)

                loss_relax = cfg.get('loss_relax', 0.1)
                accepted_indices = []
                for k_idx in range(len(seeds)):
                    if (avg_scores[k_idx] > sim_ref_min) and ((avg_losses_k[k_idx] - LB) < loss_relax):
                        accepted_indices.append(k_idx)

                if len(accepted_indices) > 0:
                    best_k = int(np.argmax(avg_scores[accepted_indices]))
                    best_k = accepted_indices[best_k]
                else:
                    best_k = int(np.argmax(avg_scores))
                    if not ((avg_losses_k[best_k] - LB) < loss_relax):
                        best_k = None

                if best_k is not None:
                    best_seed = seeds[best_k]
                    server_param_list = [p for p in self.model.parameters() if p.requires_grad]
                    
                    flat_theta = parameters_to_vector(server_param_list).detach().to(self.device)
                    delta = make_uniform_delta_from_seed(best_seed, flat_theta.cpu(), cfg['rho'], device=self.device, scale_by_norm=cfg.get('scale_by_norm'))
                    
                    beta = cfg.get('beta', 1.0)
                    new_flat = (flat_theta.to(self.device) + beta * delta.to(self.device)).clone()
                    
                    offset = 0
                    for p, n in zip(server_param_list, [p.numel() for p in server_param_list]):
                        seg = new_flat[offset: offset + n].view_as(p.data)
                        p.data.copy_(seg)
                        offset += n
                    applied = True
                else:
                    applied = False

                min_sim = sim_ref_min
                mean_sim = sim_ref_mean


            elif cfg['enable_dampening'] and (rnd >= cfg.get("D_start") and rnd < cfg.get("P_start")):
                aggregation = AgreementWeightedFedAvg(verbose=cfg.get('agg_verbose', False))
            
                server_param_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
                param_numels = {name: int(p.numel()) for name, p in self.model.named_parameters() if p.requires_grad}
            
                gradient_dicts = []
                for c in self.clients:
                    grads_cpu = c.compute_avg_gradient(global_model=self.model, local_epochs=cfg['local_epochs'], max_batches=cfg['max_client_steps'], device=self.device)
                    
                    if not grads_cpu:
                        continue
            
                    grads_fixed = {}
                    for name in server_param_names:
                        if name in grads_cpu:
                            t = grads_cpu[name]
                            if isinstance(t, np.ndarray):
                                t = torch.from_numpy(t)
                            grads_fixed[name] = t.detach().cpu().clone()
                        else:
                            grads_fixed[name] = torch.zeros(param_numels[name], dtype=torch.float32)
            
                    class _G:
                        def __init__(self, d, cid):
                            self.gradients = d
                            self.client_id = cid
                    gradient_dicts.append(_G(grads_fixed, c.name))
            
                if len(gradient_dicts) == 0:
                    client_state_dicts = []
                    for c in self.clients:
                        sd = c.local_update(self.model, local_epochs=cfg['local_epochs'],
                                            lr=cfg['local_lr'], max_steps=cfg['max_client_steps'], use_amp=cfg['use_amp'])
                        client_state_dicts.append(sd)
                    new_sd = fedavg_from_state_dicts(client_state_dicts)
                    self.model.load_state_dict(new_sd)
            
                    pct_pruned = 0.0
                    damp_W_mean = None
                    min_sim, mean_sim = None, None
                else:
                    flat_list = []
                    for g in gradient_dicts:
                        parts = []
                        for name in server_param_names:
                            t = g.gradients.get(name)
                            if t is None:
                                parts.append(np.zeros(param_numels[name], dtype=np.float32))
                            else:
                                parts.append(t.reshape(-1).numpy())
                        flat_vec = np.concatenate(parts).astype(np.float32)
                        flat_list.append(flat_vec)
            
                    min_sim, mean_sim = pairwise_cosine_stats(flat_list)  
            
                    weighted_grad = aggregation.aggregate(gradient_dicts)  
            
                    server_lr_damp = cfg.get('server_lr') * 10
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if name in weighted_grad:
                                g_cpu = weighted_grad[name] 
                                if isinstance(g_cpu, np.ndarray):
                                    g_cpu = torch.from_numpy(g_cpu)
                                param.data.add_(-server_lr_damp * g_cpu.to(param.device))
            
                    try:
                        damp_W_mean = float(np.mean([float(v.mean()) for v in aggregation.last_agreement_weights.values()])) \
                                     if hasattr(aggregation, 'last_agreement_weights') and aggregation.last_agreement_weights else None
                    except Exception:
                        damp_W_mean = None
            
                    pct_pruned = 0.0
            
            elif cfg['enable_pruning'] and (rnd >= cfg.get("P_start") and rnd <= cfg.get("rounds")):
                aggregation = PruningFedAvg(threshold=cfg.get('P_tolerance'), patience=cfg.get('P_patience'))
            
                gradient_dicts = []
                param_order = None
                for c in self.clients:
                    grads_cpu = c.compute_avg_gradient(
                        global_model=self.model,
                        local_epochs=cfg.get('local_epochs', 1),
                        max_batches=cfg.get('max_client_steps', None),
                        device=self.device
                    )
                    if not grads_cpu:
                        continue
                    if param_order is None:
                        param_order = list(grads_cpu.keys())
                    class _G:
                        def __init__(self, d, cid):
                            self.gradients = d
                            self.client_id = cid
                    gradient_dicts.append(_G(grads_cpu, c.name))
            
                if len(gradient_dicts) == 0:
                    client_state_dicts = []
                    for c in self.clients:
                        sd = c.local_update(self.model, local_epochs=cfg['local_epochs'],
                                            lr=cfg['local_lr'], max_steps=cfg['max_client_steps'], use_amp=cfg['use_amp'])
                        client_state_dicts.append(sd)
                    new_sd = fedavg_from_state_dicts(client_state_dicts)
                    self.model.load_state_dict(new_sd)
                    pct_pruned = 0.0
                    damp_W_mean = None
                    min_sim, mean_sim = None, None
                else:
                    flat_list = []
                    for g in gradient_dicts:
                        parts = []
                        for name in param_order:
                            arr = g.gradients.get(name)
                            if arr is None:
                                parts.append(np.zeros(0, dtype=float))   # unlikely, fallback
                            else:
                                parts.append(arr.reshape(-1).numpy())
                        flat_list.append(np.concatenate(parts))
                        
                    min_sim, mean_sim = pairwise_cosine_stats(flat_list)
            
                    pruned_grads, pruning_rate = aggregation.aggregate(gradient_dicts)
            
                    server_lr_prune = cfg.get('server_lr', 1e-3) * 10
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if name in pruned_grads:
                                g_cpu = pruned_grads[name]   # CPU tensor
                                param.data.add_(-server_lr_prune * g_cpu.to(self.device))
            
                    pct_pruned = pruning_rate
                    try:
                        agree_stats = aggregation.get_agreement_statistics()
                        damp_W_mean = np.mean([s['mean'] for s in agree_stats.values()]) if len(agree_stats) > 0 else None
                    except Exception:
                        damp_W_mean = None
            
            else:
                flat_server = parameters_to_vector([p for p in self.model.parameters() if p.requires_grad]).detach().cpu().numpy()
                
                for c in self.clients:
                    sd = c.local_update(self.model, cfg['local_epochs'], cfg['local_lr'], cfg['max_client_steps'], cfg['use_amp'])
                    client_state_dicts.append(sd)
                    
                    client_model = deepcopy(self.model)
                    client_model.load_state_dict(sd)
                    flat_client = parameters_to_vector([p for p in client_model.parameters() if p.requires_grad]).detach().cpu().numpy()
                    client_flat_updates.append(flat_client - flat_server)
                
                min_sim, mean_sim = pairwise_cosine_stats(client_flat_updates)
                self.model.load_state_dict(fedavg_from_state_dicts(client_state_dicts))
                pct_pruned, damp_W_mean = 0.0, None

            accs = [c.eval_on_test(self.model) for c in self.clients]
            avg_acc = float(np.mean(accs))

            self.log.append({
                'round': rnd,
                'avg_client_acc': avg_acc,
                'min_pairwise_sim': min_sim,
                'mean_pairwise_sim': mean_sim,
                'applied_delta': bool(applied),
                'applied_delta': applied,
                'pct_pruned': pct_pruned,
                'damp_W_mean': damp_W_mean,
                'time': time.time() - t0,
                'extra_grad_evals_est': extra_grad_evals
            })

            if rnd%5==0:
                print(f"[R{rnd}] avg_acc={avg_acc:.4f} min_sim={min_sim} mean_sim={mean_sim} applied_delta={applied}")

        return self.log