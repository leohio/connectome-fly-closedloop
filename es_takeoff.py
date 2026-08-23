#!/usr/bin/env python
"""統合39改訂2: 離陸→巡航のプロファイルと実回路相当の感覚 (回路模型+遅延2) で
帰還則をES再学習する。統合31のES-K (bioflight_best) から温間スタート。
適応度 = 生存 + 0.5·直立 − 0.05·巡航高度誤差 (3外乱/脚押し出しあり)。
"""
import json, os, sys
import numpy as np
from multiprocessing import Pool
os.environ.setdefault("SENSOR_DELAY", "3")
import reward_K as RK

RK.SENSOR = "circuit_model"; RK.FREQ_REFLEX = 0.0; RK.JUMP_VZ = 3.0
TH0 = np.load("outputs/es_takeoff_best.npy") if os.path.exists(
    "outputs/es_takeoff_best.npy") else np.load("outputs/bioflight_best.npy")
N_PAR = 40
Z0, ZC, RAMP = 0.6, 12.0, 3.0
T_EVAL = 6.0
N_PAIRS, SIGMA, LR, N_GEN = 10, 0.03, 0.25, 100


def zfn(t):
    tf = t - 0.4
    if tf < 0: return Z0
    if tf < RAMP: return Z0 + tf / RAMP * (ZC - Z0)
    return ZC


def fitness(th_k):
    th = np.concatenate([TH0[:12], th_k])
    tot = 0.0
    for s in range(3):
        RK.DEBUG_LOG = []
        srv, up, ze, _om = RK.rollout_sensed(th, None, T=T_EVAL, z_fn=zfn,
                                        hold_until=0.4, z_floor=0.05,
                                        start_z=Z0, clamp_x0=False,
                                        noise_seed=s)
        zl = [r[1] for r in RK.DEBUG_LOG if r[0] > RAMP + 0.8]
        zerr = float(np.mean(np.abs(np.array(zl) - ZC))) if zl else 12.0
        ups = [r[4] for r in RK.DEBUG_LOG if r[0] > 0.4]
        upmin = min(ups) if ups else -1.0
        # 安定飛行の基準S1 (背中が上: up>=0.5) を適応度に直接入れる
        tot += srv + 0.5 * max(up, 0) - 0.05 * min(zerr, 12.0) \
            + 1.0 * max(upmin, -1.0)
    return tot / 3


def work(th):
    try:
        return fitness(np.asarray(th))
    except Exception:
        return 0.0


if __name__ == "__main__":
    rg = np.random.default_rng(0)
    th = TH0[12:].copy()
    best, best_th = -1e9, th.copy()
    with Pool(12) as pool:
        for g in range(N_GEN):
            eps = rg.standard_normal((N_PAIRS, N_PAR))
            cands = [th + SIGMA * e for e in eps] + [th - SIGMA * e for e in eps]
            f = np.array(pool.map(work, cands))
            fp, fm = f[:N_PAIRS], f[N_PAIRS:]
            grad = ((fp - fm)[:, None] * eps).mean(0) / (2 * SIGMA)
            th = th + LR * grad / (np.linalg.norm(grad) + 1e-9) * SIGMA * 3
            fc = work(th)
            if fc > best:
                best, best_th = fc, th.copy()
                np.save("outputs/es_takeoff_d3_best.npy",
                        np.concatenate([TH0[:12], best_th]))
            print(f"gen{g}: 現在{fc:.3f} 最良{best:.3f} 集団max{f.max():.3f}",
                  flush=True)
    print("DONE", flush=True)
