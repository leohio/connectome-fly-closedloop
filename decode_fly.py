#!/usr/bin/env python
"""統合29 LoopH: 較正復号による閉ループ飛行。

経路: 実ハルテア求心性 → 実MANC配線 → 操舵MNの活動位相 (伝達利得1.0を実測)
      → 較正復号 ω_est = Δφ @ pinv(S) → 復元トルク -K·ω_est
      → ヒンジ有効性行列 B で翅チャネルへ配分。
S も B も「較正」であり、配線図は一切改変していない。
"""
import sys
import numpy as np
from multiprocessing import Pool


def make_decode():
    z = np.load("outputs/decode_S.npz")
    S, R, names, phi0 = z["S"], z["R"], list(z["names"]), z["phi0"]
    good = [i for i in range(len(names)) if R[i] > 0.5]
    Sg = np.linalg.pinv(S[good])        # (3, n_good) → Δφ@Sg.T? 下で整合
    Sg = np.linalg.pinv(S[good]).T      # Δφ(n,) @ Sg(n,3) = ω(3,)
    B = np.load("outputs/ctrl_B.npy")
    Bpinv = np.linalg.pinv(B)           # a(3,) @ Bpinv(3,4) = u(4,)
    jg = [names[i] for i in good]
    phg = [phi0[i] for i in good]
    return (Sg, phg, jg, Bpinv)


def one(job):
    K, seed = job
    import locked_circuit as LC
    z = np.load("outputs/locked_fit.npz")
    gc, names2, means = z["g"], list(z["names"]), z["means"]
    dec = make_decode()
    s, up = LC.fly(gc, names2, means, mode="closed", T=3.0,
                   axis_gain=(0.0015,) * 3, decode=dec, K=K,
                   pert_seed=seed)
    return s, up


if __name__ == "__main__":
    KS = [(30.0,) * 3, (100.0,) * 3, (300.0,) * 3, (800.0,) * 3]
    jobs = [(k, s) for k in KS for s in range(2)]
    with Pool(min(len(jobs), 8)) as p:
        res = p.map(one, jobs)
    for i, k in enumerate(KS):
        r = res[2 * i:2 * i + 2]
        sv = np.mean([x[0] for x in r])
        uv = np.mean([x[1] for x in r])
        print(f"K={k[0]:6.0f}: 生存{sv:.2f}s 直立{uv:+.2f} "
              f"(各{r[0][0]:.2f},{r[1][0]:.2f})", flush=True)
    print("DONE", flush=True)
