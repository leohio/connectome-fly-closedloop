#!/usr/bin/env python
"""統合29 LoopD: 反射の伝達特性をシステム同定する。

固定ωを注入した状態でlocked回路を回し、読み出しu1-4が
ωにどう応答するか (du/dω, 符号と大きさ) を直接測る。
盲目的ESの代わりに、反射ゲインを解析的に決めるための土台。
"""
import numpy as np
from brian2 import ms as _ms, Hz, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF
import locked_circuit as LC


def probe(om_fix, T=0.6, t_skip=0.25, rec_gain=None):
    """ωを固定注入して回路を回し、u1-4の平均と周期内変動を返す"""
    z = np.load("outputs/locked_fit.npz")
    gc, names2, means = z["g"], list(z["names"]), z["means"]
    M = CF.chan_matrix()
    jmap = [CF.NAMES.index(mu) for mu in names2]
    kw = dict(LC.SETUP_KW)
    if rec_gain is not None:
        kw["recruit"] = True
    net, mon, pg, pref, side, st_idx, n = PR.setup(**kw)
    ch_idx = {mu: st_idx[mu] for mu in names2 if mu in st_idx}
    us = {mu: 0.0 for mu in names2}
    fs = {mu: 0.0 for mu in names2}
    prev = 0
    shift_prev = np.zeros(len(pref))
    o = np.clip(np.asarray(om_fix, float), -12, 12)
    cR = cP = cY = PR.C_PHASE
    NBB = 40
    acc = np.zeros((NBB, 4))
    cnt = np.zeros(NBB)
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        shift = (cR * side * o[0] + cP * o[1]
                 + cY * side * np.cos(2 * np.pi * pref) * o[2])
        pg.v = pg.v - (shift - shift_prev)
        shift_prev = shift
        if rec_gain is not None:
            kR, kP, kY = rec_gain
            drv = (kR * side * o[0] + kP * o[1]
                   + kY * side * np.cos(2 * np.pi * pref) * o[2])
            PR.REC_PG.rates = np.clip(drv, 0, 600) * Hz
        net.run(CF.DT_N * 1000 * _ms)
        nsp = mon.num_spikes
        if nsp > prev:
            for iN in np.array(mon.i[prev:nsp]):
                for mu, im in ch_idx.items():
                    if int(iN) == im:
                        us[mu] += 1.0 / CF.TAU_A
            prev = nsp
        u14 = np.zeros(4)
        for jj, mu in enumerate(names2):
            us[mu] -= CF.DT_N * us[mu] / CF.TAU_A
            fs[mu] += CF.DT_N * (us[mu] - fs[mu]) / CF.TAU_A
            u14 += gc[jj] * (fs[mu] - means[jj]) * M[jmap[jj]]
        if tn > t_skip:
            b = min(int(((tn * PR.WBF) % 1.0) * NBB), NBB - 1)
            acc[b] += np.clip(u14, -0.55, 0.55)
            cnt[b] += 1
    prof = acc / np.maximum(cnt, 1)[:, None]
    return prof


if __name__ == "__main__":
    PR.C_PHASE = 0.03
    base = probe((0, 0, 0))
    print("u1-4 周期平均 (ω=0):", np.round(base.mean(0), 4), flush=True)
    print("u1-4 周期内p-p (ω=0):",
          np.round(base.max(0) - base.min(0), 4), flush=True)
    for ax, nm in [(0, "roll"), (1, "pitch"), (2, "yaw")]:
        for w in (+8.0, -8.0):
            om = [0.0, 0.0, 0.0]
            om[ax] = w
            p = probe(tuple(om))
            d = p - base
            print(f"{nm}{w:+.0f}: Δ平均={np.round(d.mean(0),4)} "
                  f"Δp-p={np.round((p.max(0)-p.min(0))-(base.max(0)-base.min(0)),4)}",
                  flush=True)
