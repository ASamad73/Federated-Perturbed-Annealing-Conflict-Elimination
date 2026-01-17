import os
import time
import argparse
import torch
import pandas as pd
import numpy as np

from src import (
    SmallCNN, 
    build_cifar_clients, 
    build_pacs_clients,
    FedClient, 
    FedGGAServer, 
    set_global_seed
)

def eval_global_on_test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="FedPACE: Federated Learning Research Framework")

    # --- General Configuration ---
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results/fed_pace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "pacs"])
    parser.add_argument("--pacs_heldout", type=str, default="sketch", help="Domain to withhold for testing")

    # --- Data & local Training ---
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--local_lr", type=float, default=1e-3)
    parser.add_argument("--server_lr", type=float, default=1e-3)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=100)

    # --- Stage 1: GGA (Annealing) ---
    parser.add_argument("--enable_gga", action="store_true", default=True)
    parser.add_argument("--k_seeds", type=int, default=8, help="K candidates for GGA")
    parser.add_argument("--rho", type=float, default=1e-5, help="Perturbation radius")
    parser.add_argument("--r_start", type=int, default=2, help="GGA start round")
    parser.add_argument("--r_end", type=int, default=15, help="GGA end round")

    # --- Stage 2: Dampening (Conflict Elimination) ---
    parser.add_argument("--enable_dampening", action="store_true", default=True)
    parser.add_argument("--d_start", type=int, default=20, help="Round to start conflict dampening")
    parser.add_argument("--beta", type=float, default=0.3, help="Dampening scaling factor")

    # --- Stage 3: Pruning (Elimination) ---
    parser.add_argument("--enable_pruning", action="store_true", default=True)
    parser.add_argument("--p_start", type=int, default=42, help="Round to start pruning")
    parser.add_argument("--p_tolerance", type=float, default=0.2, help="Agreement threshold for pruning")
    parser.add_argument("--p_patience", type=int, default=1, help="Rounds to wait before pruning")

    # --- Advanced Settings ---
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Mixed Precision")
    parser.add_argument("--loss_relax", type=float, default=0.05, help="GGA loss relaxation threshold")
    parser.add_argument("--scale_by_norm", action="store_true", default=True)

    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = {
        'rounds': args.rounds,
        'R_start': args.r_start,
        'R_end': args.r_end,
        'D_start': args.d_start,
        'P_start': args.p_start,
        'P_tolerance': args.p_tolerance,
        'P_patience': args.p_patience,
        'K': args.k_seeds,
        'rho': args.rho,
        'beta': args.beta,
        'local_epochs': args.local_epochs,
        'max_client_steps': args.max_steps,
        'local_lr': args.local_lr,
        'server_lr': args.server_lr,
        'enable_gga': args.enable_gga,
        'enable_dampening': args.enable_dampening,
        'enable_pruning': args.enable_pruning,
        'use_amp': args.use_amp,
        'scale_by_norm': args.scale_by_norm,
        'loss_relax': args.loss_relax
    }

    # 1. Set Reproducibility
    set_global_seed(args.seed)

    # 2. Build Data Partitions (From src/data.py)
    results = []

    # --- CASE 1: PACS (Leave-One-Domain-Out Rotation) ---
    if args.dataset == "pacs":
        print("--- Initiating PACS Rotation (Leave-One-Out) ---")
        # build_pacs_clients now returns a DICT of all domain loaders
        domain_loaders = build_pacs_clients(batch_size=args.batch_size)
        domains = list(domain_loaders.keys())

        for held_out in domains:
            train_domains = [d for d in domains if d != held_out]
            print(f"\n>> Held-out Domain: {held_out.upper()} | Training on: {train_domains}")
            
            set_global_seed(args.seed)
            clients = [FedClient(d, domain_loaders[d], domain_loaders[d], device) for d in train_domains]
            
            model = SmallCNN(num_classes=7).to(device)
            held_test_loader = domain_loaders[held_out]
            
            server = FedGGAServer(model, clients, device, cfg, test_loader=held_test_loader)
            run_log = server.run()

            held_client = FedClient(held_out, None, held_test_loader, device)
            acc = held_client.eval_on_test(server.model)
            
            results.append({'held_out': held_out, 'acc': acc, 'dataset': 'PACS'})
            print(f"Accuracy for {held_out}: {acc:.4f}")

    # --- CASE 2: CIFAR-10 (Standard Dirichlet Non-IID) ---
    else:
        print(f"--- Running CIFAR-10 (Alpha: {args.alpha}) ---")
        set_global_seed(args.seed)
        clients_list, test_loader, _ = build_cifar_clients(
            root="./data", alpha=args.alpha, batch_size=args.batch_size
        )
        clients = [FedClient(name, loader, test_loader, device) for name, loader in clients_list]
        
        model = SmallCNN(num_classes=10).to(device)
        server = FedGGAServer(model, clients, device, cfg, test_loader=test_loader)
        run_log = server.run()
        
        acc = eval_global_on_test(server.model, test_loader, device) # Using your existing helper
        results.append({'held_out': 'N/A', 'acc': acc, 'dataset': 'CIFAR10'})

    # --- Final Result Saving ---
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.save_dir, f"{args.dataset}_final_results.csv"), index=False)
    print(f"\nFinal Summary:\n{df}")
    if args.dataset == "pacs":
        print(f"Average PACS Accuracy: {df['acc'].mean():.4f}")

if __name__ == "__main__":
    main()