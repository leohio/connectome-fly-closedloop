#!/usr/bin/env python
"""統合20: 多筋位相デコード — 行動レベルの配線特異性への挑戦。

統合19の結論: b1振幅1チャネルの位相反射は行動レベルでシャッフルと分離できない
(非特異的ディザと同等)。実物は約12本の操舵筋が別々の位相窓で発火し
振幅・ストローク中心・迎角を多重制御する。本統合ではデコーダを多筋化し、
「読み出し帯域を増やせば配線特異性が行動に現れるか」を検証する。

mode=prc_full : 実配線の全操舵筋PRC (どのチャネルが使えるか)
mode=sweep    : REALでゲイン粗探索 (n=3)
mode=main     : n=8本実験。対照は SHUF に加え REAL-位相スクランブル
                (同じ回路・同じスパイク統計で基準位相の対応のみ破壊
                 = ディザ交絡を直接分離する)
"""
import sys
import numpy as np
import phase_reflex as PR
from shuffle_control import circ_diff, CONDS


def calibrate(shuffle=None, seed=0, T=0.4):
    """ω=0 で全操舵筋の基準位相を測る。{mu: (rate, ph0, R)} を返す。"""
    cal = PR.prc_run(shuffle=shuffle, shuffle_seed=seed, omegas=(0.0,), T=T)
    return {mu: v for (om, mu), v in cal.items()}


def mode_prc_full():
    print("=== 実配線: 全操舵筋PRC (ω_roll = 0, +20, -20) ===", flush=True)
    res = PR.prc_run(omegas=(0.0, 20.0, -20.0), T=0.5)
    mus = sorted({mu for (_, mu) in res})
    print(f"{'筋':8s} {'rate':>6s} {'R':>5s} {'位相0':>6s} "
          f"{'Δ(+20)':>8s} {'Δ(-20)':>8s} {'方向感度':>8s}")
    for mu in mus:
        v0 = res.get((0.0, mu))
        if v0 is None:
            print(f"{mu:8s} 無発火")
            continue
        vp = res.get((20.0, mu)); vm = res.get((-20.0, mu))
        dp = circ_diff(vp[1], v0[1]) if vp else np.nan
        dm = circ_diff(vm[1], v0[1]) if vm else np.nan
        dirs = (dm - dp) / 2 if np.isfinite(dp) and np.isfinite(dm) else np.nan
        print(f"{mu:8s} {v0[0]:6.1f} {v0[2]:5.2f} {v0[1]:6.3f} "
              f"{dp:+8.3f} {dm:+8.3f} {dirs:+8.3f}", flush=True)
    np.savez("outputs/multi_prc.npz",
             **{f"{om}|{mu}": np.array(v) for (om, mu), v in res.items()})


def mode_sweep():
    PR.C_PHASE = 0.012
    cal = calibrate()
    PH0 = {mu: v[1] for mu, v in cal.items()}
    print(f"較正チャネル数: {len(PH0)}", flush=True)
    for Gm in [dict(amp=5.0, hg=1.5, iii=1.2),
               dict(amp=10.0, hg=3.0, iii=2.5),
               dict(amp=20.0, hg=6.0, iii=5.0)]:
        ll = []
        for rep in range(3):
            s, up, nst, upl = PR.fly_trial(0.0, PH0=PH0, decode="multi",
                                           G_multi=Gm)
            ll.append(upl)
            print(f"  amp={Gm['amp']} hg={Gm['hg']} iii={Gm['iii']} "
                  f"rep{rep}: 生存{s:.2f}s 後半{upl:+.3f} st={nst}", flush=True)
        print(f"Gm={Gm}: 後半直立度 {np.mean(ll):+.3f}±{np.std(ll):.3f}",
              flush=True)


def mode_main():
    PR.C_PHASE = 0.012
    Gm = eval(sys.argv[2]) if len(sys.argv) > 2 else dict(amp=10.0, hg=3.0,
                                                          iii=2.5)
    NREP = 8
    print(f"=== 統合20本実験 (n={NREP}, Gm={Gm}) ===", flush=True)
    s0, up0, _, upl0 = PR.fly_trial(0.0)
    print(f"{'反射OFF':14s} 生存{s0:.2f}s 後半直立度{upl0:+.3f}", flush=True)
    rows = [("OFF", [s0], [upl0], [0])]
    conds = [("REAL", None, 0, False), ("SCRAM", None, 0, True),
             ("SHUF-HAL s1", "hal", 1, False),
             ("SHUF-ALL s0", "all", 0, False)]
    for name, sh, seed, scram in conds:
        cal = calibrate(shuffle=sh, seed=seed)
        PH0 = {mu: v[1] for mu, v in cal.items()}
        if scram:   # 基準位相の筋への対応をシャッフル (回路・スパイクは実物のまま)
            rng = np.random.default_rng(7)
            keys = sorted(PH0)
            vals = rng.permutation([PH0[k] for k in keys])
            PH0 = dict(zip(keys, vals))
        ss, ll, bb = [], [], []
        for rep in range(NREP):
            s, up, nst, upl = PR.fly_trial(0.0, PH0=PH0, decode="multi",
                                           G_multi=Gm, shuffle=sh,
                                           shuffle_seed=seed)
            ss.append(s); ll.append(upl); bb.append(nst)
            print(f"  {name} rep{rep}: 生存{s:.2f}s 後半{upl:+.3f} st={nst}",
                  flush=True)
        sem = np.std(ll) / np.sqrt(len(ll))
        print(f"{name:14s} 生存{np.mean(ss):.2f}±{np.std(ss):.2f}s "
              f"後半直立度{np.mean(ll):+.3f}±SEM{sem:.3f} "
              f"st={np.mean(bb):.0f}", flush=True)
        rows.append((name, ss, ll, bb))
    np.savez("outputs/multi_main.npz", rows=np.array(rows, dtype=object))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "prc_full"
    {"prc_full": mode_prc_full, "sweep": mode_sweep,
     "main": mode_main}[mode]()
