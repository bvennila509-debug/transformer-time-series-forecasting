#!/usr/bin/env python3
"""generate_multivariate_ts.py
Create a synthetic multivariate time series with trend, multiple seasonalities, noise, and two regressors.
Outputs a CSV with datetime index and columns y (target), reg1, reg2.
"""
import argparse, numpy as np, pandas as pd

def create_time_index(n_periods=1200, freq='D', start='2015-01-01'):
    return pd.date_range(start=start, periods=n_periods, freq=freq)

def synth_multivariate(n_periods=1200, seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(n_periods)
    trend = 0.02 * t + 0.00005 * (t**1.9)
    yearly = 8.0 * np.sin(2 * np.pi * t / 365.25)
    weekly = 2.0 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(scale=2.5, size=n_periods)
    y = (trend + (yearly + weekly)) + noise
    # regressors
    reg1 = (rng.rand(n_periods) > 0.96).astype(int)  # promo-like binary
    reg2 = 20 + 5 * np.sin(2 * np.pi * t / 365.25) + rng.normal(scale=1.0, size=n_periods)  # temp-like
    df = pd.DataFrame({'ds': create_time_index(n_periods, freq), 'y': y, 'reg1': reg1, 'reg2': reg2})
    return df

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--n_periods', type=int, default=1200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--freq', default='D')
    args = p.parse_args()
    df = synth_multivariate(n_periods=args.n_periods, seed=args.seed)
    df.to_csv(args.out, index=False)
    print('Saved', args.out)
