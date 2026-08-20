#!/usr/bin/env python
"""統合29 LoopG: 神経によるω測定行列 S を較正する。

S[mu, ax] = d(操舵筋muの活動位相)/d(ω_ax)。
実ハルテア求心性→実MANC配線→操舵MNという経路だけで決まる量。
これを較正すれば、回路の筋位相からωを復号できる (ω_est = Δφ @ pinv(S))。
シャッフル配線ではSが変わるので、行動レベルの配線特異性の検定に使える。
"""
import sys
import numpy as np
from brian2 import ms as _ms, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF
import locked_circuit as LC   # WBF/C_PHASE の生理設定を継承

MUS = ["b1_L", "b1_R", "b2_R", "i1_L", "i1_R", "i2_L",
       "iii1_L", "iii1_R", "iii3_R", "hg1_R", "hg3_R", "hg4_R"]
W_PROBE = 10.0
T = 0.5
T0 = 0.15


def phases(om, shuffle=None, seed=0):
    kw = dict(LC.SETUP_KW)
    if shuffle:
        kw["shuffle"] = shuffle
        kw["shuffle_seed"] = seed
    net, mon, pg, pref, side, st_idx, n = PR.setup(**kw)
    o = np.clip(np.asarray(om, float), -12, 12)
    shift = PR.C_PHASE * (side * o[0] + o[1]
                          + side * np.cos(2 * np.pi * pref) * o[2])
    pg.v = pg.v - shift
    # This is a neural-only calibration with a constant angular-velocity
    # condition.  Unlike the flight loop, no body state is exchanged every
    # 0.1 ms, so repeatedly calling ``run`` only recompiles/checks the same
    # schedule 5,000 times.  A single continuous run is mathematically
    # identical for the autonomous locked-afferent network and is orders of
    # magnitude faster.
    net.run(T * 1000 * _ms, namespace={})
    tr = mon.spike_trains()
    out = {}
    for mu in MUS:
        if mu not in st_idx:
            continue
        st = np.array(tr[st_idx[mu]] / _ms) / 1000.0
        st = st[st > T0]
        if len(st) < 3:
            continue
        z = np.mean(np.exp(2j * np.pi * np.mod(st * PR.WBF, 1.0)))
        out[mu] = (np.angle(z) / (2 * np.pi), abs(z))
    return out


def build(shuffle=None, seed=0, tag=""):
    base = phases((0, 0, 0), shuffle, seed)
    S = np.zeros((len(MUS), 3))
    Rok = np.zeros(len(MUS))
    for ax in range(3):
        om_p = [0.0] * 3
        om_p[ax] = W_PROBE
        om_m = [0.0] * 3
        om_m[ax] = -W_PROBE
        pp = phases(tuple(om_p), shuffle, seed)
        mm = phases(tuple(om_m), shuffle, seed)
        for j, mu in enumerate(MUS):
            if mu not in pp or mu not in mm or mu not in base:
                continue
            dp = (pp[mu][0] - base[mu][0] + 0.5) % 1.0 - 0.5
            dm = (mm[mu][0] - base[mu][0] + 0.5) % 1.0 - 0.5
            S[j, ax] = (dp - dm) / (2 * W_PROBE)     # 奇成分のみ採用
            Rok[j] = min(pp[mu][1], mm[mu][1], base[mu][1])
    return S, Rok, base


if __name__ == "__main__":
    S, Rok, base = build()
    print("S [cycle/(rad/s)] 行=筋, 列=roll,pitch,yaw", flush=True)
    for j, mu in enumerate(MUS):
        print(f"  {mu:8s} {S[j,0]:+.5f} {S[j,1]:+.5f} {S[j,2]:+.5f}  "
              f"(R={Rok[j]:.2f})", flush=True)
    good = Rok > 0.5
    print(f"\n位相固定が十分な筋: {good.sum()}/{len(MUS)}", flush=True)
    Sg = S[good]
    if Sg.shape[0] >= 3:
        sv = np.linalg.svd(Sg, compute_uv=False)
        print(f"特異値: {np.round(sv, 5)}  条件数="
              f"{sv[0]/max(sv[-1],1e-12):.1f}", flush=True)
    np.savez("outputs/decode_S.npz", S=S, R=Rok,
             names=np.array(MUS),
             phi0=np.array([base.get(mu, (np.nan, 0))[0] for mu in MUS]))
    print("保存 outputs/decode_S.npz", flush=True)
    print("DONE", flush=True)
