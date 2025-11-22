#!/usr/bin/env python3
"""train_transformer.py
Train and evaluate the Transformer forecaster on synthetic or provided CSV data.
"""
import argparse, os, time, json
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformer_forecaster import TransformerForecaster
import utils as ut

def create_dataset_from_csv(path, seq_len=90, pred_len=30, features=None):
    df = pd.read_csv(path, parse_dates=['ds'] if 'ds' in pd.read_csv(path, nrows=0).columns else None)
    # ensure columns: either ['ds','y', ...] or ['y', ...]
    if 'y' not in df.columns:
        raise ValueError('CSV must contain column "y" as target.')
    cols = ['y'] + [c for c in df.columns if c not in ['ds','y']]
    arr = df[cols].values.astype('float32')  # shape (T, features)
    # normalize (simple)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-6
    arr = (arr - mean) / std
    X, Y = [], []
    for x,y in ut.sliding_windows(arr, seq_len, pred_len):
        X.append(x)
        Y.append(y)
    X = np.stack(X)  # (N, seq_len, feat)
    Y = np.stack(Y)  # (N, pred_len)
    # train/val/test split
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    datasets = {
        'train': (X[:train_end], Y[:train_end]),
        'val': (X[train_end:val_end], Y[train_end:val_end]),
        'test': (X[val_end:], Y[val_end:])
    }
    norm = {'mean': mean.tolist(), 'std': std.tolist()}
    return datasets, norm

def train_epoch(model, opt, loader, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = torch.nn.functional.mse_loss(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--seq_len', type=int, default=90)
    p.add_argument('--pred_len', type=int, default=30)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--output', default='results')
    args = p.parse_args()

    datasets, norm = create_dataset_from_csv(args.data, seq_len=args.seq_len, pred_len=args.pred_len)
    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # small model for demonstration
    input_dim = datasets['train'][0].shape[2]
    model = TransformerForecaster(input_dim=input_dim, d_model=64, nhead=4, num_layers=2, pred_len=args.pred_len)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # dataloaders
    def make_loader(xy):
        x,y = xy
        return DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=args.batch, shuffle=True)
    train_loader = make_loader(datasets['train'])
    val_loader = make_loader(datasets['val'])
    test_loader = make_loader(datasets['test'])

    best_val = float('inf')
    history = {'train_loss':[], 'val_loss':[]}
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_epoch(model, opt, train_loader, device)
        preds_val, trues_val = eval_model(model, val_loader, device)
        val_loss = ((preds_val - trues_val) ** 2).mean()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pt'))
        print(f'Epoch {epoch} | train_loss {train_loss:.6f} | val_loss {val_loss:.6f} | time {time.time()-t0:.1f}s')

    # final evaluation on test set
    preds_test, trues_test = eval_model(model, test_loader, device)
    # denormalize using saved stats
    mean = np.array(norm['mean'])[0,0] if isinstance(norm['mean'][0], list) else np.array(norm['mean'])[0]
    std = np.array(norm['std'])[0,0] if isinstance(norm['std'][0], list) else np.array(norm['std'])[0]
    # note: target column was first column in normalization
    preds_test_denorm = preds_test * std[0] + mean[0]
    trues_test_denorm = trues_test * std[0] + mean[0]
    import utils as ut
    test_rmse = ut.rmse(trues_test_denorm.flatten(), preds_test_denorm.flatten())
    test_mae = ut.mae(trues_test_denorm.flatten(), preds_test_denorm.flatten())
    metrics = {'test_rmse': float(test_rmse), 'test_mae': float(test_mae)}
    with open(os.path.join(args.output,'metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)
    print('Test RMSE:', test_rmse, 'Test MAE:', test_mae)
    print('Saved metrics to', os.path.join(args.output,'metrics.json'))

if __name__ == '__main__':
    main()
