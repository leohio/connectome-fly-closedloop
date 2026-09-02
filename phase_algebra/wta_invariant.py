#!/usr/bin/env python
"""P3 変種C: 勝者総取り (winner-take-all) の条件数不変量。

実配線の実測 S の各行は ±C_PHASE の符号パターン (求心性クラスの符号化ベクトル) に
一致する: MN の発火位相は入力の重心ではなく、最強の (電気結合した) 求心性 1 本の
位相に従う。したがって復号行列の条件数は、MN 群が代表する求心性クラス
(側 ±1 × 好み位相クラスタ 0/½ → 符号化ベクトル [side, 1, side·cos2πpref]) の
多様性で決まる。構造から:

    j*(i) = argmax_j W_ij         (W = 直接化学 + 電気; 電気 8mV が支配的)
    S_wta[i, :] = C_PHASE · [side_j*, 1, side_j* cos(2π pref_j*)]
    invariant = cond(S_wta)  (良い MN = 最大入力が十分強いもの)

これを実測の条件数・復号誤差 (100 ヌル) と突き合わせる。
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import phase_algebra.wiring_operator as WO
import phase_reflex as PR


def wta_S(ex, min_w=4.0):
    W = np.clip(WO.effective_W(ex, hops=0), 0, None)
    js = W.argmax(1); top = W.max(1)
    good = top >= min_w
    pref, side = ex["pref"], ex["side"]
    rows = np.array([[side[j], 1.0, side[j] * np.cos(2 * np.pi * pref[j])] for j in js]) * PR.C_PHASE
    return rows, good, js


def main():
    dq = {(r["tag"], r["seed"]): r for r in json.load(open("outputs/decode_quality_full_null.json"))}
    rows = []
    for tag, seed in [("real", 0)] + [("shuffle", s) for s in range(100)]:
        ex = WO.extract("all" if tag == "shuffle" else None, seed)
        S, good, js = wta_S(ex)
        Sg = S[good]
        sv = np.linalg.svd(Sg, compute_uv=False) if good.sum() >= 3 else np.array([1, 0, 0])
        cls = set((int(ex["side"][j]), int(round(ex["pref"][j] * 2)) % 2) for j in js[good])
        m = dq.get((tag, seed), {})
        rows.append(dict(tag=tag, seed=seed, cond_wta=float(sv[0] / max(sv[-1], 1e-12)), rank=int((sv > 1e-9 * sv[0]).sum()),
                         n_good=int(good.sum()), n_class=len(cls), cond_meas=m.get("cond", np.nan), err_meas=m.get("err", np.nan)))
    json.dump(rows, open("phase_algebra/outputs/wta_invariant.json", "w"), indent=1)
    real = rows[0]; sh = [r for r in rows[1:] if np.isfinite(r["err_meas"])]
    def rank_(v): o = np.argsort(v); rr = np.empty(len(v)); rr[o] = np.arange(len(v)); return rr
    x = np.log(np.array([min(r["cond_wta"], 1e6) for r in sh])); ym = np.log(np.array([r["cond_meas"] for r in sh])); ye = np.log(np.array([r["err_meas"] for r in sh]))
    print(f"実配線: WTA条件数 {real['cond_wta']:.2f} (クラス数 {real['n_class']}, 良MN {real['n_good']}) / 実測条件数 {real['cond_meas']:.2f} / 実測誤差 {real['err_meas']:.3f}")
    print(f"ヌル: WTA条件数 中央値 {np.median([r['cond_wta'] for r in sh]):.2f} / クラス数分布 {dict(zip(*np.unique([r['n_class'] for r in sh], return_counts=True)))}")
    print(f"WTA条件数 → 実測条件数: Spearman ρ={np.corrcoef(rank_(x), rank_(ym))[0,1]:+.3f}")
    print(f"WTA条件数 → 実測復号誤差: Spearman ρ={np.corrcoef(rank_(x), rank_(ye))[0,1]:+.3f}")
    nc = np.array([r["n_class"] for r in sh]); e = np.array([r["err_meas"] for r in sh])
    for c in sorted(set(nc)): print(f"  クラス数={c}: ヌル{(nc==c).sum():3d}個  実測誤差 中央値 {np.median(e[nc==c]):.3f}")
    print(f"実配線のWTA条件数はヌル{len(sh)}個中 {(np.array([r['cond_wta'] for r in sh]) <= real['cond_wta']).sum()} 個より悪い/同等")


if __name__ == "__main__":
    main()
