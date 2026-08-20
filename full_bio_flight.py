#!/usr/bin/env python
"""統合36b 最終検証: 適応要素をすべて生物学的学習で獲得した個体を、実回路で飛ばす。

構成 (適応要素に外部最適化なし):
  生得 (進化相当・固定): 翅運動学7+トリム5、回路の生理定数
  局所デルタ則で獲得: 読出しW (統合35、教師=視覚由来の粗いω)
  報酬変調学習で獲得: 帰還則K+b (統合36、報酬=ハエが感知できる量のみ)
  感覚: 実MANCハルテア求心性→実配線→操舵MN→筋活動位相 (統合31)

これが通れば「実コネクトーム + 2つの生物学的学習則 + 生得波形」の飛行が成立する。
"""
import sys
import json
import numpy as np
from multiprocessing import Pool


def job(a):
    which, pert = a
    import numpy as np
    import plastic_readout as P
    import connectome_bioflight as CB
    import reward_K as RK
    res = json.load(open("outputs/reward_K.json"))
    best = max(res, key=lambda r: r["final"])
    th_learned = np.array(best["theta"])
    # 帰還則を学習品へ差し替え (K_POL/B_POL は connectome_bioflight のモジュール変数)
    if which in ("learnedK", "full_bio"):
        CB.K_POL = th_learned[:CB.N_U * CB.N_X].reshape(CB.N_U, CB.N_X)
        CB.B_POL = th_learned[CB.N_U * CB.N_X:]
    if which in ("learnedW", "full_bio"):      # 読出しも学習品
        d = P.episodes(rng_seed=0, n_train=120)
        W, _ = P.learn(d, rng_seed=0, checkpoints=[120])
        dec = (W, d["names"], d["phi0"])
    else:                                       # 読出しは外部較正
        dec = CB.calibrate()
    r = CB.fly(src="circuit", dec=dec, T=10.0, pert_seed=pert)
    return (which, pert, *r)


if __name__ == "__main__":
    modes = ["full_bio", "learnedK", "calibrated"]
    jobs = [(m, p) for m in modes for p in [0, 1]]
    with Pool(len(jobs)) as pool:
        out = pool.map(job, jobs)
    lab = {"full_bio": "全適応要素が生物学的学習 (K=報酬学習, W=局所則)",
           "learnedK": "K=報酬学習, W=外部較正",
           "calibrated": "K=ES, W=外部較正 (従来)"}
    for m in modes:
        v = [r for r in out if r[0] == m]
        sv = np.array([x[2] for x in v])
        up = np.array([x[3] for x in v])
        ze = np.array([x[4] for x in v])
        print(f"{lab[m]}: 生存{sv.mean():5.2f}s 直立{up.mean():+.2f} "
              f"高度誤差{ze.mean():.1f} ({', '.join(f'{x:.2f}' for x in sv)})",
              flush=True)
    print("DONE", flush=True)
