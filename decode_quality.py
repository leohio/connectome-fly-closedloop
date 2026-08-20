#!/usr/bin/env python
"""統合32/33: 神経ω復号の質を、実配線とfull-nullで統計比較する。

これまでのシャッフル対照は構成ごとに復号Sを再較正していたため、
配線を変えても較正が吸収し、行動レベルでは差が出なかった。
ここでは「較正の機会を等しく与えたうえで、復号の質そのもの」を測る:

- 位相固定できる筋の数 (R>0.5)
- S の条件数と感度の大きさ
- **較正と異なるωでの復号誤差** (線形性・汎化の検定)

統合32当時のshuffleは化学配線だけを置換し、補完したハルテア→MN電気経路を
実配線のまま残していた (partial-null)。統合33では phase_reflex.setup が
化学・電気の両方を次数保存で置換する。旧結果は
outputs/decode_quality_partial_null.json に保存し、本スクリプトの出力は
full-nullとして別名にする。

実配線がfull-null分布の外に出れば、それは行動に効きうる水準の配線特異性である。
"""
import sys
import numpy as np
from multiprocessing import Pool

W_CAL = 10.0
TESTS = [(4.0, 0, 0), (0, 4.0, 0), (0, 0, 4.0),
         (3.0, -3.0, 2.0), (-6.0, 2.0, -4.0)]


def analyse(arg):
    tag, seed = arg
    import numpy as np
    import measure_S as MS
    import connectome_bioflight as CB       # 翅周波数の設定を継承
    MS.PR.WBF = CB.PR.WBF
    sh = "all" if tag == "shuffle" else None
    S, R, base = MS.build(shuffle=sh, seed=seed)
    good = [i for i in range(len(MS.MUS)) if R[i] > 0.5]
    if len(good) < 3:
        return dict(tag=tag, seed=seed, n_good=len(good), cond=np.nan,
                    sens=np.nan, err=np.nan)
    Sg = S[good]
    sv = np.linalg.svd(Sg, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-12))
    sens = float(np.linalg.norm(Sg))
    Dec = np.linalg.pinv(Sg).T              # Δφ @ Dec = ω
    errs = []
    for om in TESTS:
        p = MS.phases(om, shuffle=sh, seed=seed)
        dphi = []
        okall = True
        for i in good:
            mu = MS.MUS[i]
            if mu not in p or mu not in base:
                okall = False
                break
            dd = (p[mu][0] - base[mu][0] + 0.5) % 1.0 - 0.5
            dphi.append(dd)
        if not okall:
            continue
        est = np.array(dphi) @ Dec
        errs.append(np.linalg.norm(est - np.array(om))
                    / max(np.linalg.norm(om), 1e-9))
    return dict(tag=tag, seed=seed, n_good=len(good), cond=cond,
                sens=sens, err=float(np.mean(errs)) if errs else np.nan)


if __name__ == "__main__":
    n_sh = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    jobs = [("real", 0)] + [("shuffle", s) for s in range(n_sh)]
    with Pool(min(len(jobs), 11)) as p:
        res = p.map(analyse, jobs)
    real = [r for r in res if r["tag"] == "real"][0]
    shuf = [r for r in res if r["tag"] == "shuffle"]
    print(f"実配線 : 固定筋{real['n_good']:2d}  条件数{real['cond']:6.2f}  "
          f"感度{real['sens']:.5f}  復号誤差{real['err']:.3f}", flush=True)
    for r in shuf:
        print(f"シャッフル{r['seed']}: 固定筋{r['n_good']:2d}  "
              f"条件数{r['cond']:6.2f}  感度{r['sens']:.5f}  "
              f"復号誤差{r['err']:.3f}", flush=True)
    for key, name in [("n_good", "位相固定できる筋数"),
                      ("cond", "条件数(小さいほど良い)"),
                      ("sens", "感度の大きさ"),
                      ("err", "復号誤差(小さいほど良い)")]:
        v = np.array([r[key] for r in shuf], float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        rv = real[key]
        z = (rv - v.mean()) / max(v.std(), 1e-12)
        pct = float((v < rv).mean() * 100)
        print(f"{name:22s} 実={rv:8.4f}  シャッフル={v.mean():8.4f}"
              f"±{v.std():.4f}  z={z:+.2f}  上位{100-pct:.0f}%", flush=True)
    import json
    json.dump(res, open("outputs/decode_quality_full_null.json", "w"),
              default=float)
    # 片側の経験的p値 (実配線がシャッフルより復号誤差が小さい確率)
    ev = np.array([r["err"] for r in shuf], float)
    ev = ev[np.isfinite(ev)]
    n_better = int((ev <= real["err"]).sum())
    print(f"\n片側経験的p値 (復号誤差): {(n_better + 1) / (len(ev) + 1):.4f} "
          f"(シャッフル{len(ev)}個中{n_better}個が実配線以下)", flush=True)
    print("保存 outputs/decode_quality_full_null.json", flush=True)
    print("DONE", flush=True)
