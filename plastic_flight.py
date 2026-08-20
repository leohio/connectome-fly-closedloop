#!/usr/bin/env python
"""統合35b: 局所可塑性で獲得した読出しだけで飛ぶ (外部較正ゼロの飛行)。

これまでの飛行 (統合31-34) は読出しSを外部最小二乗で較正していた。
ここでは plastic_readout の局所デルタ則 (教師=視覚由来の粗いω) が獲得したWを
そのまま connectome_bioflight の飛行ループに差し込み、外部較正と比較する。

成立すれば: 実MANC配線 + 局所学習則 + 動物が持つ教師信号 だけで
読出しが立ち上がり、そのまま10秒級の飛行を支えられることになる。
52次元の帰還則Kは依然として外部最適化であることは明記する (残る非生物学化点)。
"""
import sys
import numpy as np
from multiprocessing import Pool


def job(a):
    mode, pert = a
    import plastic_readout as P
    import connectome_bioflight as CB
    if mode.startswith("learned"):
        n_ep = int(mode.split("_")[1])
        d = P.episodes(rng_seed=0, n_train=n_ep)
        W, _ = P.learn(d, rng_seed=0, checkpoints=[n_ep])
        dec = (W, d["names"], d["phi0"])
    else:                                   # 外部較正 (従来)
        dec = CB.calibrate()
    r = CB.fly(src="circuit", dec=dec, T=10.0, pert_seed=pert)
    return (mode, pert, *r)


if __name__ == "__main__":
    perts = [0, 1]
    modes = ["learned_120", "learned_40", "calibrated"]
    jobs = [(m, p) for m in modes for p in perts]
    with Pool(len(jobs)) as pool:
        res = pool.map(job, jobs)
    for m in modes:
        v = [r for r in res if r[0] == m]
        sv = np.array([x[2] for x in v])
        up = np.array([x[3] for x in v])
        ze = np.array([x[4] for x in v])
        lab = {"learned_120": "局所学習 (120エピソード)",
               "learned_40": "局所学習 ( 40エピソード)",
               "calibrated": "外部較正 (従来)"}[m]
        print(f"{lab}: 生存{sv.mean():5.2f}s 直立{up.mean():+.2f} "
              f"高度誤差{ze.mean():.1f} "
              f"({', '.join(f'{x:.2f}' for x in sv)})", flush=True)
    print("DONE", flush=True)
