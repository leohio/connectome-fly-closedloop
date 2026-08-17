#!/usr/bin/env python
"""統合29 最終ベンチ: 生物学的に利用可能な感覚だけで何ができるかを比較する。

教師の3.0sは「自分の羽ばたき反動振動を完全ジャイロで読む」ことに依存し、
実バエのハルテアが出せる信号ではないことが判明した。よって公平な比較は
「体の回転(遅い成分)しか使えない条件」で行う。
"""
import sys
import numpy as np
from multiprocessing import Pool

N_SEED = 4
T = 3.0


def one(job):
    cond, seed = job
    if cond.startswith("teacher_"):
        import gyro_swap as GS
        return GS.trial(src=cond[len("teacher_"):], T=T, pert_seed=seed)[0]
    import locked_circuit as LC
    z = np.load("outputs/locked_fit.npz")
    gc, names2, means = z["g"], list(z["names"]), z["means"]
    if cond == "circuit_ff":
        return LC.fly(gc, names2, means, mode="ff", T=T, pert_seed=seed)[0]
    if cond == "circuit_closed":
        return LC.fly(gc, names2, means, mode="closed", T=T,
                      pert_seed=seed)[0]
    if cond == "circuit_shuffle":
        return LC.fly(gc, names2, means, mode="closed", T=T, pert_seed=seed,
                      shuffle="all", shuffle_seed=seed)[0]
    raise ValueError(cond)


if __name__ == "__main__":
    conds = sys.argv[1:]
    jobs = [(c, s) for c in conds for s in range(N_SEED)]
    with Pool(min(len(jobs), 12)) as p:
        res = p.map(one, jobs)
    k = 0
    for c in conds:
        v = np.array(res[k:k + N_SEED])
        k += N_SEED
        print(f"{c:18s} 生存 平均{v.mean():.2f}s ± {v.std():.2f}  "
              f"({', '.join(f'{x:.2f}' for x in v)})", flush=True)
    print("DONE", flush=True)
