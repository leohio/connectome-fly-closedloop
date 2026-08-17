#!/usr/bin/env python
"""統合29: 運動学習 — 生理学的パラメタのみの調律で実回路閉ループ飛行を伸ばす。

学習対象 (27次元、すべて生物が発達・経験で調律するパラメタ):
- 12 筋張力ゲイン (NMJ強度・筋断面積、log空間)
- 12 ヒンジ位相ゲイン (操舵筋の活動位相→翅運動学の変換利得;
  Tu & Dickinson 1996 のヒンジ・ギアボックスに対応)
- 3 軸別ハルテア反射感度 (生理基準 C_PHASE=0.0015 に対する倍率)
固定 (学習しない): MANC配線、locked透過段、電気シナプス、AHP、LIFパラメタ、
チャネル写像、教師波形。配線図そのものは一切いじらない。

評価は複数の初期外乱の平均 (単発はカオス的で分散が大きい)。
"""
import json
import os
import sys
import numpy as np

CKPT = "outputs/motor_learn.json"
NP = 12
SIG_G = 0.12      # log筋ゲイン
SIG_H = 1.0       # ヒンジ位相ゲイン (Δφ~0.012周期なので h~8 が適正域;
                  # σ=8では±16まで振れて出力飽和し全候補が不安定化した)
SIG_A = 0.3       # 軸別反射感度 (倍率)
LR = 0.5
T_EVAL = 3.0
N_PAIRS = 5
N_PERT = 2        # 評価あたりの初期外乱数
WORKERS = 10


def unpack(theta):
    gmul = np.exp(theta[:NP])
    h = theta[NP:2 * NP]
    ax = np.clip(theta[2 * NP:2 * NP + 3], -3, 3)
    return gmul, h, ax


def evaluate(theta):
    import locked_circuit as LC
    import phase_reflex as PR
    z = np.load("outputs/locked_fit.npz")
    gc0, names2, means = z["g"], list(z["names"]), z["means"]
    phi0 = z["phi0"]
    gmul, h, ax = unpack(theta)
    base = 0.0015
    ag = tuple(base * (1.0 + a) for a in ax)
    srv, ups = [], []
    for s in range(N_PERT):
        r = LC.fly(gc0 * gmul, names2, means, mode="closed", T=T_EVAL,
                   axis_gain=ag, phase_gain=h, phi0=phi0, pert_seed=s)
        srv.append(r[0])
        ups.append(r[1])
    return (float(np.mean(srv)) + 0.3 * max(float(np.mean(ups)), 0.0),
            float(np.mean(srv)), float(np.mean(ups)))


def worker(args):
    _, theta = args
    try:
        return evaluate(np.asarray(theta))
    except Exception as e:
        print("worker error:", repr(e), flush=True)
        return (0.0, 0.0, 0.0)


def main():
    from multiprocessing import Pool
    rng = np.random.default_rng(0)
    dim = 2 * NP + 3
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        theta = np.array(ck["theta"])
        if theta.size != dim:
            theta = np.zeros(dim)
            gen0, best, best_theta = 0, -1.0, theta.copy()
        else:
            gen0 = ck["gen"]
            best = ck["best"]
            best_theta = np.array(ck["best_theta"])
    else:
        theta = np.zeros(dim)
        gen0, best, best_theta = 0, -1.0, theta.copy()
    sig = np.array([SIG_G] * NP + [SIG_H] * NP + [SIG_A] * 3)
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    with Pool(WORKERS) as pool:
        for gen in range(gen0, gen0 + n_gen):
            eps = rng.standard_normal((N_PAIRS, dim))
            cands = []
            for e in eps:
                cands.append(theta + sig * e)
                cands.append(theta - sig * e)
            res = pool.map(worker, list(enumerate(cands)))
            fits = np.array([r[0] for r in res])
            grad = np.zeros(dim)
            for k, e in enumerate(eps):
                grad += (fits[2 * k] - fits[2 * k + 1]) * e
            grad /= (2 * N_PAIRS)
            theta = theta + LR * sig * grad
            gi = int(np.argmax(fits))
            if fits[gi] > best:
                best = float(fits[gi])
                best_theta = np.array(cands[gi])
            print(f"gen{gen}: best={best:.3f} genbest={fits[gi]:.3f} "
                  f"mean={fits.mean():.3f} "
                  f"srv={','.join(f'{r[1]:.2f}' for r in res)}", flush=True)
            json.dump({"gen": gen + 1, "theta": theta.tolist(),
                       "best": best, "best_theta": best_theta.tolist()},
                      open(CKPT, "w"))
    f0, s0, u0 = evaluate(theta)
    fb, sb, ub = evaluate(best_theta)
    print(f"final center: 生存{s0:.2f}s 直立{u0:+.2f}", flush=True)
    print(f"final best  : 生存{sb:.2f}s 直立{ub:+.2f}", flush=True)


if __name__ == "__main__":
    main()
