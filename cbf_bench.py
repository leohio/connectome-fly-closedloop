#!/usr/bin/env python
"""統合31 最終ベンチ: コネクトーム駆動の生物学的飛行を複数外乱で検証する。"""
import sys
import numpy as np
from multiprocessing import Pool

N_SEED = 4


def job(a):
    cond, seed = a
    import connectome_bioflight as CB
    if cond in ("circuit", "shuffle"):
        dec = CB.calibrate(shuffle="all" if cond == "shuffle" else None,
                           seed=seed)
        return CB.fly(src=cond, dec=dec, pert_seed=seed,
                      shuffle="all" if cond == "shuffle" else None, seed=seed)
    return CB.fly(src=cond, pert_seed=seed)


if __name__ == "__main__":
    conds = sys.argv[1:] or ["true", "circuit", "shuffle", "zero"]
    jobs = [(c, s) for c in conds for s in range(N_SEED)]
    with Pool(min(len(jobs), 16)) as p:
        res = p.map(job, jobs)
    k = 0
    for c in conds:
        v = res[k:k + N_SEED]
        k += N_SEED
        sv = np.array([x[0] for x in v])
        up = np.array([x[1] for x in v])
        ze = np.array([x[2] for x in v])
        print(f"{c:9s} 生存 平均{sv.mean():.2f}s ±{sv.std():.2f} "
              f"直立{up.mean():+.2f} 高度誤差{ze.mean():.1f} "
              f"({', '.join(f'{x:.2f}' for x in sv)})", flush=True)
    print("DONE", flush=True)
