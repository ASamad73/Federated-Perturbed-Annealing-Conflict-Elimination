import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_runlog_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Runlog not found: {path}")
    df = pd.read_csv(path)
    if 'round' in df.columns:
        df['round'] = df['round'].astype(int)
        df = df.sort_values('round').reset_index(drop=True)
    return df

def select_accuracy_series(df, acc_priority=('global_accu', 'avg_client_acc', 'val_acc')):
    acc_col = None
    for c in acc_priority:
        if c in df.columns:
            acc_col = c
            break
    if acc_col is None:
        raise ValueError(f"No accuracy column found in runlog. Expected one of {acc_priority}")
    s = pd.to_numeric(df[acc_col], errors='coerce')
    s = s.ffill().bfill()
    if 'round' in df.columns:
        s.index = df['round'].values
    else:
        s.index = np.arange(len(s))
    return s, acc_col

def select_similarity_series(df, sim_col='mean_pairwise_sim'):
    if sim_col in df.columns:
        s = pd.to_numeric(df[sim_col], errors='coerce')
        s = s.ffill().bfill()
        if 'round' in df.columns:
            s.index = df['round'].values
        else:
            s.index = np.arange(len(s))
    else:
        n = len(df)
        idx = df['round'].values if 'round' in df.columns else np.arange(n)
        s = pd.Series([np.nan]*n, index=idx)
    return s

def compute_metrics_from_series(acc_s, sim_s):
    if len(acc_s) == 0:
        raise ValueError("Empty accuracy series.")
    final_acc = float(acc_s.iloc[-1])
    mean_sim = float(sim_s.dropna().mean()) if sim_s.dropna().size > 0 else float('nan')
    
    rounds = np.asarray(acc_s.index, dtype=float)
    if len(rounds) >= 2:
        auc_raw = np.trapz(y=acc_s.values, x=rounds)
        denom = (rounds[-1] - rounds[0]) if (rounds[-1] - rounds[0]) > 0 else 1.0
        auc = float(auc_raw / denom)
    else:
        auc = float(acc_s.iloc[-1])
    return {'final_acc': final_acc, 'mean_pairwise_sim': mean_sim, 'auc': auc}

def plot_from_runlog(runlog_path, out_fig=None, title=None, show=True):
    df = load_runlog_csv(runlog_path)
    acc_s, acc_col = select_accuracy_series(df)
    sim_s = select_similarity_series(df, sim_col='mean_pairwise_sim')

    metrics = compute_metrics_from_series(acc_s, sim_s)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    rounds = acc_s.index.values

    ax1.plot(rounds, acc_s.values, marker='o', linewidth=2, label=f'Accuracy ({acc_col})')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if np.all(np.isnan(sim_s.values)):
        ax2.text(0.5, 0.5, 'No Similarity Data', transform=ax2.transAxes, ha='center', alpha=0.5)
        ax2.set_ylabel('Similarity (N/A)')
    else:
        ax2.plot(rounds, sim_s.values, marker='x', linestyle='--', linewidth=1.5, color='tab:orange', label='Gradient Similarity')
        ax2.set_ylabel('Pairwise Cosine Similarity')
        ax2.set_ylim(bottom=0, top=1.0) 

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')

    if title is None:
        title = os.path.basename(runlog_path)
    plt.title(title)
    plt.tight_layout()

    if out_fig:
        os.makedirs(os.path.dirname(out_fig), exist_ok=True)
        plt.savefig(out_fig, dpi=300)
        print(f"Plot saved to: {out_fig}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Metrics: Final Acc: {metrics['final_acc']:.4f} | AUC: {metrics['auc']:.4f} | Mean Sim: {metrics['mean_pairwise_sim']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from FedPACE runlogs")
    parser.add_argument("--log", type=str, required=True, help="Path to the .csv runlog file")
    parser.add_argument("--out", type=str, default=None, help="Path to save the .png figure")
    parser.add_argument("--title", type=str, default=None, help="Title for the plot")
    parser.add_argument("--no_show", action="store_true", help="Don't display the plot window")
    
    args = parser.parse_args()

    if args.out is None:
        args.out = args.log.replace(".csv", ".png")

    plot_from_runlog(args.log, out_fig=args.out, title=args.title, show=not args.no_show)