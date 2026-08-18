#!/usr/bin/env python
"""統合31: 3秒は評価窓の上限だったので、10秒で真に安定かを確認する。"""
import sys
import numpy as np
from multiprocessing import Pool


def job(a):
    cond, seed = a
    import connectome_bioflight as CB
    dec = None
    if cond in ("circuit", "shuffle"):
        dec = CB.calibrate(shuffle="all" if cond == "shuffle" else None,
                           seed=seed)
    return CB.fly(src=cond, dec=dec, T=10.0, pert_seed=seed,
                  shuffle="all" if cond == "shuffle" else None, seed=seed)


if __name__ == "__main__":
    conds = sys.argv[1:] or ["true", "circuit"]
    jobs = [(c, s) for c in conds for s in range(2)]
    with Pool(min(len(jobs), 8)) as p:
        res = p.map(job, jobs)
    k = 0
    for c in conds:
        v = res[k:k + 2]
        k += 2
        sv = np.array([x[0] for x in v])
        up = np.array([x[1] for x in v])
        ze = np.array([x[2] for x in v])
        print(f"{c:9s} 10秒試験: 生存 平均{sv.mean():.2f}s 直立{up.mean():+.2f} "
              f"高度誤差{ze.mean():.1f} ({', '.join(f'{x:.2f}' for x in sv)})",
              flush=True)
    print("DONE", flush=True)
