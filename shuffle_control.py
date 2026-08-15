#!/usr/bin/env python
"""統合19: シャッフルコネクトーム対照実験。

問い: 位相反射(統合18)の効果は「実配線だから」か?
対照: 次数保存シャッフル(エッジの重みとpre端を保持し、post端のみ置換)
  - SHUF-HAL: ハルテア求心性の出力エッジのみ置換
    → ハルテア→b1の787シナプス集中と左右特異性を破壊
  - SHUF-ALL: サブ回路の全エッジを置換
    → 反射弧の構造全体を破壊 (総興奮量・次数分布は保存)

mode=smoke : 各条件でω=0のb1発火を確認 (回路が生きているか)
mode=prc   : 位相応答曲線の比較 (位相伝送がシャッフルで消えるか)
mode=fly   : 飛行テスト (G=20, C_PHASE=0.012, 条件別PH0較正)
"""
import sys
import numpy as np
import phase_reflex as PR


def circ_diff(a, b):
    """位相差 a-b を [-0.5, 0.5) cycle で返す"""
    return (a - b + 0.5) % 1.0 - 0.5


CONDS = [("REAL", None, 0),
         ("SHUF-HAL s0", "hal", 0), ("SHUF-HAL s1", "hal", 1),
         ("SHUF-ALL s0", "all", 0), ("SHUF-ALL s1", "all", 1)]


def mode_smoke():
    for name, sh, seed in [("REAL", None, 0), ("SHUF-HAL s0", "hal", 0),
                           ("SHUF-ALL s0", "all", 0)]:
        res = PR.prc_run(shuffle=sh, shuffle_seed=seed, omegas=(0.0,), T=0.3)
        rows = {mu: res.get((0.0, mu)) for mu in ["b1_L", "b1_R"]}
        msg = " ".join(
            f"{mu}: 無発火" if v is None else
            f"{mu}: {v[0]:.0f}Hz 位相{v[1]:.3f} R={v[2]:.2f}"
            for mu, v in rows.items())
        print(f"{name:12s} {msg}", flush=True)


def mode_prc():
    all_res = {}
    print("=== PRC比較 (ω_roll = 0, +20, -20 rad/s) ===", flush=True)
    for name, sh, seed in CONDS:
        res = PR.prc_run(shuffle=sh, shuffle_seed=seed)
        all_res[name] = res
        for mu in ["b1_L", "b1_R"]:
            v0 = res.get((0.0, mu))
            vp = res.get((20.0, mu))
            vm = res.get((-20.0, mu))
            if v0 is None:
                print(f"{name:12s} {mu}: ω=0で無発火", flush=True)
                continue
            dp = circ_diff(vp[1], v0[1]) if vp else np.nan
            dm = circ_diff(vm[1], v0[1]) if vm else np.nan
            trans = np.nanmean([abs(dp), abs(dm)])
            print(f"{name:12s} {mu}: rate={v0[0]:6.1f}Hz R={v0[2]:.2f} "
                  f"位相0={v0[1]:.3f} Δ(+20)={dp:+.3f} Δ(-20)={dm:+.3f} "
                  f"|伝送|={trans:.3f}", flush=True)
    np.savez("outputs/shuffle_prc.npz",
             **{f"{name}|{om}|{mu}": np.array(v)
                for name, res in all_res.items()
                for (om, mu), v in res.items()})


def mode_fly():
    PR.C_PHASE = 0.012
    G = 20.0
    print("=== 飛行テスト (G=20, C_PHASE=0.012, 各条件PH0較正) ===", flush=True)
    s0, up0, _, _ = PR.fly_trial(0.0)
    print(f"{'反射OFF':12s} 生存{s0:.2f}s 直立度{up0:+.2f}", flush=True)
    out = [("OFF", s0, up0, 0)]
    for name, sh, seed in CONDS:
        cal = PR.prc_run(shuffle=sh, shuffle_seed=seed, omegas=(0.0,), T=0.4)
        vL = cal.get((0.0, "b1_L")); vR = cal.get((0.0, "b1_R"))
        if vL is None and vR is None:
            print(f"{name:12s} b1無発火 → 反射は構造的に不能", flush=True)
            PH0 = None
        else:
            PH0 = {"b1_L": vL[1] if vL else 0.5, "b1_R": vR[1] if vR else 0.5}
        s, up, nb1, _ = PR.fly_trial(G, PH0=PH0, shuffle=sh, shuffle_seed=seed)
        ph_s = ("-" if PH0 is None
                else f"PH0=({PH0['b1_L']:.2f},{PH0['b1_R']:.2f})")
        print(f"{name:12s} 生存{s:.2f}s 直立度{up:+.2f} b1spk={nb1} {ph_s}",
              flush=True)
        out.append((name, s, up, nb1))
    np.savez("outputs/shuffle_fly.npz", rows=np.array(out, dtype=object))


def mode_fly2():
    """リーク付き反射(スパイクが来なければ消灯)で各条件3反復。
    反射OFFは神経が翅に影響しないため決定論的 → 1回のみ。"""
    PR.C_PHASE = 0.012
    G = 20.0
    NREP = 3
    print("=== 飛行テスト v2 (リーク付き, 3反復, G=20) ===", flush=True)
    s0, up0, _, upl0 = PR.fly_trial(0.0)
    print(f"{'反射OFF':12s} 生存{s0:.2f}s 直立度{up0:+.2f}", flush=True)
    rows = [("OFF", [s0], [up0], [0])]
    for name, sh, seed in CONDS:
        cal = PR.prc_run(shuffle=sh, shuffle_seed=seed, omegas=(0.0,), T=0.4)
        vL = cal.get((0.0, "b1_L")); vR = cal.get((0.0, "b1_R"))
        PH0 = {"b1_L": vL[1] if vL else 0.5, "b1_R": vR[1] if vR else 0.5}
        ss, uu, bb = [], [], []
        for rep in range(NREP):
            s, up, nb1, _ = PR.fly_trial(G, PH0=PH0, shuffle=sh,
                                         shuffle_seed=seed)
            ss.append(s); uu.append(up); bb.append(nb1)
            print(f"  {name} rep{rep}: 生存{s:.2f}s 直立度{up:+.2f} "
                  f"b1={nb1}", flush=True)
        print(f"{name:12s} 生存{np.mean(ss):.2f}±{np.std(ss):.2f}s "
              f"直立度{np.mean(uu):+.2f}±{np.std(uu):.2f} "
              f"b1={np.mean(bb):.0f}", flush=True)
        rows.append((name, ss, uu, bb))
    np.savez("outputs/shuffle_fly2.npz", rows=np.array(rows, dtype=object))


def mode_fly3():
    """検出力を上げた最終比較: 主要指標=反射作動区間(t>=0.12)の直立度、
    n=8/条件。SHUF-HAL s0 (b1構造的沈黙≈OFF) は除外し4条件。"""
    PR.C_PHASE = 0.012
    G = 20.0
    NREP = 8
    conds = [("REAL", None, 0), ("SHUF-HAL s1", "hal", 1),
             ("SHUF-ALL s0", "all", 0), ("SHUF-ALL s1", "all", 1)]
    print("=== 飛行テスト v3 (主要指標=後半直立度, n=8) ===", flush=True)
    s0, up0, _, upl0 = PR.fly_trial(0.0)
    print(f"{'反射OFF':12s} 生存{s0:.2f}s 後半直立度{upl0:+.3f}", flush=True)
    rows = [("OFF", [s0], [upl0], [0])]
    for name, sh, seed in conds:
        cal = PR.prc_run(shuffle=sh, shuffle_seed=seed, omegas=(0.0,), T=0.4)
        vL = cal.get((0.0, "b1_L")); vR = cal.get((0.0, "b1_R"))
        PH0 = {"b1_L": vL[1] if vL else 0.5, "b1_R": vR[1] if vR else 0.5}
        ss, ll, bb = [], [], []
        for rep in range(NREP):
            s, up, nb1, upl = PR.fly_trial(G, PH0=PH0, shuffle=sh,
                                           shuffle_seed=seed)
            ss.append(s); ll.append(upl); bb.append(nb1)
            print(f"  {name} rep{rep}: 生存{s:.2f}s 後半{upl:+.3f} "
                  f"b1={nb1}", flush=True)
        sem = np.std(ll) / np.sqrt(len(ll))
        print(f"{name:12s} 生存{np.mean(ss):.2f}±{np.std(ss):.2f}s "
              f"後半直立度{np.mean(ll):+.3f}±SEM{sem:.3f} "
              f"b1={np.mean(bb):.0f}", flush=True)
        rows.append((name, ss, ll, bb))
    np.savez("outputs/shuffle_fly3.npz", rows=np.array(rows, dtype=object))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    {"smoke": mode_smoke, "prc": mode_prc, "fly": mode_fly,
     "fly2": mode_fly2, "fly3": mode_fly3}[mode]()
