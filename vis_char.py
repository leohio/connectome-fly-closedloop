#!/usr/bin/env python
"""統合37a: 脳視覚経路 (オセリ+VS/HS→OCG/中継/DN) の姿勢復号の応答特性を測る。"""
import numpy as np
from brian2 import Hz, ms as _ms, prefs
prefs.codegen.target = "numpy"
import vision_pop as VP
import brain_visual as BV

DT_W = 0.002

def characterize(tau_dn):
    VP.TAU_DN = tau_dn
    net, mon, pg, ssign, is_oc, readout = VP.build()
    r0, t_r, t_p, s2r, s2p = VP.calibrate(net, mon, pg, ssign, is_oc, readout)
    prev = mon.count[:].copy()
    f_rate = r0.copy()
    # ステップ応答: t=0.2でroll 0→0.15rad
    log_t, log_est = [], []
    for k in range(int(0.8 / DT_W)):
        t = k * DT_W
        roll = 0.15 if t >= 0.2 else 0.0
        pg.rates = BV.vis_rates(roll, 0.0, 0, 0, ssign, is_oc) * Hz
        net.run(DT_W * 1000 * _ms)
        c = mon.count[:].copy()
        f_rate += DT_W * ((c - prev)[readout] / DT_W - f_rate) / tau_dn
        prev = c
        log_t.append(t)
        log_est.append(VP.decode(f_rate, r0, t_r, s2r))
    T = np.array(log_t); E = np.array(log_est)
    pre = E[(T > 0.05) & (T < 0.2)]
    post = E[T > 0.55]
    noise = pre.std()
    gain = post.mean() / 0.15
    # 63%到達時刻
    th63 = 0.15 * gain * 0.63
    idx = np.where((T >= 0.2) & (E >= th63))[0]
    lat63 = (T[idx[0]] - 0.2) * 1000 if len(idx) else np.nan
    print(f"τ_DN={tau_dn*1000:.0f}ms: ゲイン={gain:.2f} 63%遅延={lat63:.0f}ms "
          f"静止ノイズstd={noise:.4f}rad ({np.degrees(noise):.2f}°) "
          f"SNR@0.15rad={0.15*gain/max(noise,1e-9):.1f}", flush=True)

for tau in [0.03, 0.05, 0.1]:
    characterize(tau)
print("DONE", flush=True)
