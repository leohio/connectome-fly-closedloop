#!/usr/bin/env python
"""統合32b: 課題を難しくすれば配線特異性が行動に現れるか、という予測の検証。

統合32の解釈: 行動レベルで差が出なかったのは配線に情報が無いからではなく、
飛行課題が過剰決定 (9筋で3軸) で、劣った復号でも十分だったから。
ならば復号に使える筋を3本に絞れば (3筋で3軸=ぎりぎり決定)、
実配線とシャッフルの差が飛行に現れるはずである。
"""
import sys
import numpy as np
from multiprocessing import Pool

import os
N_MUS = int(os.environ.get("N_MUS", "3"))
PERT = float(os.environ.get("PERT", "1.5"))
N_PERT = int(os.environ.get("N_PERT", "2"))


def job(a):
    tag, seed, pert = a
    import connectome_bioflight as CB
    CB.PERT_SIG = PERT
    sh = "all" if tag == "shuffle" else None
    try:
        dec = CB.calibrate(shuffle=sh, seed=seed,
                           n_mus=None if N_MUS <= 0 else N_MUS)
        r = CB.fly(src="circuit", dec=dec, T=3.0, pert_seed=pert,
                   shuffle=sh, seed=seed)
        return (tag, seed, pert, r[0], r[1], r[2])
    except Exception as e:
        return (tag, seed, pert, 0.0, 0.0, 20.0)


if __name__ == "__main__":
    n_sh = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [("real", 0, p) for p in range(N_PERT)]
    jobs += [("shuffle", s, p) for s in range(n_sh) for p in range(N_PERT)]
    with Pool(min(len(jobs), 14)) as p:
        res = p.map(job, jobs)
    real = [r for r in res if r[0] == "real"]
    shuf = [r for r in res if r[0] == "shuffle"]
    rv = np.array([r[3] for r in real])
    print(f"復号に使う筋: {'全部' if N_MUS <= 0 else str(N_MUS)+'本'} / "
          f"初期外乱σ={PERT} rad/s ({N_PERT}外乱平均)", flush=True)
    print(f"実配線   : 生存 平均{rv.mean():.2f}s "
          f"({', '.join(f'{x:.2f}' for x in rv)})", flush=True)
    per = {}
    for t, s, p_, sv, up, ze in shuf:
        per.setdefault(s, []).append(sv)
    means = []
    for s in sorted(per):
        v = np.array(per[s])
        means.append(v.mean())
        print(f"シャッフル{s}: 生存 平均{v.mean():.2f}s "
              f"({', '.join(f'{x:.2f}' for x in v)})", flush=True)
    means = np.array(means)
    n_ceil = int((means >= 2.99).sum())
    print(f"\n3.00s(上限)に達したシャッフル: {n_ceil}/{len(means)}", flush=True)
    n_better = int((means >= rv.mean()).sum())
    print(f"\n実配線 {rv.mean():.2f}s vs シャッフル {means.mean():.2f}"
          f"±{means.std():.2f}", flush=True)
    print(f"片側経験的p値: {(n_better + 1) / (len(means) + 1):.4f} "
          f"(シャッフル{len(means)}個中{n_better}個が実配線以上)", flush=True)
    print("DONE", flush=True)
