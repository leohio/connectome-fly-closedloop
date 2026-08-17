#!/usr/bin/env python
"""統合29: 評価プロトコルの是正 — 複数初期外乱の平均で条件を比較する。

単発試行はカオス的で分散が大きい (シャッフル対照が0.29-0.93sに散った)。
条件ごとにN個の再現可能な初期外乱で評価し、平均±標準偏差で報告する。
"""
import sys
import numpy as np
from multiprocessing import Pool

N_SEED = 5
T_EVAL = 3.0


def one(job):
    cond, seed = job
    import locked_circuit as LC
    import connectome_fastloop as CF
    z = np.load("outputs/locked_fit.npz")
    gc, names2, means = z["g"], list(z["names"]), z["means"]
    if cond == "ideal_ff":
        return CF.trial(mode="ideal_ff", T=T_EVAL)[0]
    if cond == "locked_ff":
        return LC.fly(gc, names2, means, mode="ff", T=T_EVAL,
                      pert_seed=seed)[0]
    if cond == "locked_closed":
        return LC.fly(gc, names2, means, mode="closed", T=T_EVAL,
                      pert_seed=seed)[0]
    raise ValueError(cond)


if __name__ == "__main__":
    conds = sys.argv[1:] or ["locked_ff", "locked_closed"]
    jobs = [(c, s) for c in conds for s in range(N_SEED)]
    with Pool(min(len(jobs), 10)) as p:
        res = p.map(one, jobs)
    k = 0
    for c in conds:
        v = np.array(res[k:k + N_SEED])
        k += N_SEED
        print(f"{c:16s} 生存 平均{v.mean():.2f}s ± {v.std():.2f} "
              f"(各: {', '.join(f'{x:.2f}' for x in v)})", flush=True)
