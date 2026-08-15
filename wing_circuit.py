#!/usr/bin/env python
"""(b') 飛行状態の翅運動回路: DLMn緊張性発火 + ハルテア反射の検証。

構成:
  - DLMn/DVMn (動力筋MN) に緊張性内因電流 → 飛行中の持続発火(実測5-20Hz)
  - ハルテア求心性 (DMetaN の proprioceptive SN) を羽ばたき位相に固定した
    レートで駆動。体回転 ω を左右差・位相変調として注入(ハルテア=ジャイロ)
  - 出力: 操舵筋MN (b1,b2,b3,i1,i2,iii1,iii3,hg1-4) の発火率と位相

検証項目:
  1. DLMn が緊張性に発火するか(実測 5-20Hz)
  2. b1 が羽ばたき位相に固定して発火するか(実物の看板現象)
  3. 体回転(roll/pitch/yaw)がどの操舵筋をどの向きに変調するか
     = コネクトーム反射行列(学習した制御則Kの実配線版)
"""
import numpy as np
import pandas as pd
from brian2 import ms, Hz, prefs

import vnc_model

prefs.codegen.target = "numpy"

WBF = 218.0
DT_WIN = 0.002
STEER = ["b1", "b2", "b3", "i1", "i2", "iii1", "iii3",
         "hg1", "hg2", "hg3", "hg4"]


def wing_tables():
    mns = pd.read_csv("../vnc-connectome/downloads/elife-96084-supp3-v1.csv",
                      encoding="latin1")
    wm = mns[mns.subclass == "wm"].copy()
    power_ids = wm[wm.target.astype(str).str.startswith(("DLM", "DVM"))
                   ].bodyid.tolist()
    steer = wm[wm.target.isin(STEER)][["bodyid", "target"]]
    props = pd.read_feather("neuron-properties.feather")
    som = props.set_index("bodyId")["somaSide"]
    steer["side"] = steer.bodyid.map(som)
    hal = props[(props["class"] == "sensory neuron")
                & (props.modality == "proprioceptive")
                & props.entryNerve.isin(["DMetaN_L", "DMetaN_R"])]
    hal_L = hal[hal.entryNerve == "DMetaN_L"].bodyId.tolist()
    hal_R = hal[hal.entryNerve == "DMetaN_R"].bodyId.tolist()
    return power_ids, steer, hal_L, hal_R


def run_trial(omega=(0.0, 0.0, 0.0), T=1.0, r_hal=80.0, gyro_gain=3.0,
              seed=0):
    """体回転 omega=(roll,pitch,yaw)[rad/s] を与えた時の翅MN応答を返す。"""
    power_ids, steer, hal_L, hal_R = wing_tables()
    tonic = {int(b): 8.5 for b in power_ids}          # DLMn/DVMn 緊張性
    ctrl = hal_L + hal_R
    net, mon, ids, idx, pg = vnc_model.make_network(
        [], r_stim_hz=0, sensory_bodyids=ctrl, tonic_mv=tonic)
    kept = [b for b in ctrl if b in idx]
    pos_of = {b: i for i, b in enumerate(kept)}
    sl_L = np.array([pos_of[b] for b in hal_L if b in pos_of])
    sl_R = np.array([pos_of[b] for b in hal_R if b in pos_of])
    rates = np.zeros(len(kept))
    wr, wp, wy = omega
    for wi in range(int(T / DT_WIN)):
        t = wi * DT_WIN
        ph = 2 * np.pi * WBF * t
        base = r_hal * 0.5 * (1 + np.cos(ph))          # 羽ばたき位相固定
        # ハルテアのジャイロ変調: roll→左右差, pitch→同相, yaw→位相シフト項
        mod_L = 1 + gyro_gain * (+0.02 * wr + 0.02 * wp * np.cos(ph)
                                 + 0.02 * wy * np.sin(ph))
        mod_R = 1 + gyro_gain * (-0.02 * wr + 0.02 * wp * np.cos(ph)
                                 - 0.02 * wy * np.sin(ph))
        rates[sl_L] = np.clip(base * mod_L, 0, 400)
        rates[sl_R] = np.clip(base * mod_R, 0, 400)
        pg.rates = rates * Hz
        net.run(2 * ms)
    trains = mon.spike_trains()
    out = {}
    for _, r in steer.iterrows():
        if r.bodyid in idx:
            st = np.array(trains[idx[r.bodyid]] / ms) / 1000.0
            key = f"{r.target}_{str(r.side)[:1]}"
            out[key] = dict(rate=len(st) / T,
                            phases=np.mod(st * WBF, 1.0))
    # 動力筋
    pw = [len(np.array(trains[idx[b]])) / T for b in power_ids if b in idx]
    return out, float(np.mean(pw))


if __name__ == "__main__":
    print("=== 静止(回転なし) ===")
    out0, dlm = run_trial()
    print(f"動力筋MN平均発火率: {dlm:.1f} Hz (実測 5-20Hz)")
    for k in sorted(out0):
        o = out0[k]
        if o["rate"] > 1:
            ph = o["phases"]
            # 位相集中度 (ベクトル強度)
            R = np.abs(np.mean(np.exp(2j * np.pi * ph))) if len(ph) else 0
            print(f"  {k:8s} {o['rate']:6.1f} Hz  位相固定度 R={R:.2f}")
    print("\n=== ロール +10 rad/s ===")
    outR, _ = run_trial(omega=(10, 0, 0))
    print("=== ピッチ +10 rad/s ===")
    outP, _ = run_trial(omega=(0, 10, 0))
    rows = []
    for k in sorted(out0):
        r0 = out0[k]["rate"]
        rr = outR.get(k, {}).get("rate", 0) - r0
        rp = outP.get(k, {}).get("rate", 0) - r0
        rows.append((k, r0, rr, rp))
    print("\n=== コネクトーム反射行列 (Δ発火率 [Hz]) ===")
    print(f"{'筋':8s} {'基準':>7s} {'Δroll':>7s} {'Δpitch':>7s}")
    for k, r0, rr, rp in rows:
        print(f"{k:8s} {r0:7.1f} {rr:+7.1f} {rp:+7.1f}")
    np.savez("outputs/wing_reflex.npz", rows=np.array(rows, dtype=object))
