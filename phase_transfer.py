#!/usr/bin/env python
"""統合29 LoopE: 筋活動位相のω応答 (奇関数性) を測る。

張力振幅の読み出しはωに対して偶関数 (整流) で符号を失っていた。
ヒンジは位相→振幅変換器なので (Tu & Dickinson 1996)、操舵に使うべき量は
各操舵筋の活動位相 φ_mu。これが ω に対して符号を持つ (奇関数) かを直接測る。
"""
import sys
import numpy as np
from brian2 import ms as _ms, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF

PR.WBF = float(PR._fly_env()["Pw"]["freq"])
SETUP_KW = dict(afferent_mode="locked", pref_mode="reversal",
                electrical=True, mn_ahp=True)
MUS = ["b1_L", "b1_R", "b2_R", "i1_L", "i1_R", "i2_L",
       "iii1_L", "iii1_R", "iii3_R", "hg1_R", "hg3_R", "hg4_R"]


def locked_phases(om, T=0.45, t0=0.15):
    """固定ωを注入して各操舵筋の活動位相と固定度を返す"""
    net, mon, pg, pref, side, st_idx, n = PR.setup(**SETUP_KW)
    o = np.clip(np.asarray(om, float), -12, 12)
    shift = PR.C_PHASE * (side * o[0] + o[1]
                          + side * np.cos(2 * np.pi * pref) * o[2])
    pg.v = pg.v - shift          # 定常ωなので一度だけ位相をずらす
    for kn in range(int(T / CF.DT_N)):
        net.run(CF.DT_N * 1000 * _ms)
    tr = mon.spike_trains()
    out = {}
    for mu in MUS:
        if mu not in st_idx:
            continue
        st = np.array(tr[st_idx[mu]] / _ms) / 1000.0
        st = st[st > t0]
        if len(st) < 3:
            continue
        z = np.mean(np.exp(2j * np.pi * np.mod(st * PR.WBF, 1.0)))
        out[mu] = (np.angle(z) / (2 * np.pi), abs(z))
    return out


if __name__ == "__main__":
    PR.C_PHASE = 0.03
    base = locked_phases((0, 0, 0))
    W = [-8.0, -4.0, 4.0, 8.0]
    for ax, nm in [(0, "roll"), (1, "pitch"), (2, "yaw")]:
        curves = {}
        for w in W:
            om = [0.0, 0.0, 0.0]
            om[ax] = w
            p = locked_phases(tuple(om))
            for mu, (phi, R) in p.items():
                if mu not in base or R < 0.5:
                    continue
                d = (phi - base[mu][0] + 0.5) % 1.0 - 0.5
                curves.setdefault(mu, {})[w] = d
        print(f"--- {nm} 軸: Δφ (筋活動位相のずれ, cycle) ---", flush=True)
        for mu, c in curves.items():
            if len(c) < 4:
                continue
            vals = [c[w] for w in W]
            odd = (vals[3] - vals[0]) / 2      # 奇成分 (符号を担う)
            even = (vals[3] + vals[0]) / 2     # 偶成分 (整流)
            print(f"  {mu:8s} ω=-8:{vals[0]:+.3f} -4:{vals[1]:+.3f} "
                  f"+4:{vals[2]:+.3f} +8:{vals[3]:+.3f} | "
                  f"奇{odd:+.3f} 偶{even:+.3f}", flush=True)
    print("DONE", flush=True)
