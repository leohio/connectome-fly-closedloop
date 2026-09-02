#!/usr/bin/env python
"""P3 変種D: 符号化ベクトルのコヒーレンス不変量。

MN の位相応答の線形性を決めるのは、入力求心性の基底位相 pref ではなく
符号化ベクトル b_j = [side_j, 1, side_j·cos(2π pref_j)] (ω で位相がどう動くか) の
MN 内での揃い方である。完全ヌルは電気経路の post 端を置換するため 1 つの MN に
左右両側の求心性が混ざり、ロール成分 (side·ω_x) が相殺して条件数が悪化・非線形化する。

    S_soft[i] = Σ_j W_ij b_j / Σ_j W_ij         (重み付き平均の符号化 = 予測 S 行)
    coh_i     = |Σ_j W_ij b_j| / Σ_j W_ij |b_j|  (∈[0,1], 1=完全に揃っている)
    不変量: cond(S_soft), mean/min coh_i, ロール成分の相殺度
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import phase_algebra.wiring_operator as WO


def invariants(ex, min_w=4.0, use_electrical_only=False):
    W = np.clip(WO.effective_W(ex, hops=0), 0, None)
    if use_electrical_only:
        W = ex["E"][ex["mn"]]
    pref, side = ex["pref"], ex["side"]
    B = np.stack([side, np.ones_like(side), side * np.cos(2 * np.pi * pref)], 1)   # (n_h, 3)
    tot = W.sum(1); good = tot >= min_w
    S_soft = (W @ B) / np.maximum(tot, 1e-12)[:, None]
    coh = np.linalg.norm(W @ B, axis=1) / np.maximum((W * np.linalg.norm(B, axis=1)[None, :]).sum(1), 1e-12)
    roll_coh = np.abs(W @ side) / np.maximum(tot, 1e-12)          # 左右混合の指標 (1=同側のみ)
    Sg = S_soft[good]
    sv = np.linalg.svd(Sg, compute_uv=False) if good.sum() >= 3 else np.array([1, 0, 0])
    return dict(cond_soft=float(sv[0] / max(sv[-1], 1e-12)), coh_mean=float(coh[good].mean()), coh_min=float(coh[good].min()),
                roll_coh_mean=float(roll_coh[good].mean()), roll_coh_min=float(roll_coh[good].min()), n_good=int(good.sum()))


def main():
    dq = {(r["tag"], r["seed"]): r for r in json.load(open("outputs/decode_quality_full_null.json"))}
    rows = []
    for tag, seed in [("real", 0)] + [("shuffle", s) for s in range(100)]:
        ex = WO.extract("all" if tag == "shuffle" else None, seed)
        r = invariants(ex); r.update(tag=tag, seed=seed)
        m = dq.get((tag, seed), {}); r["cond_meas"] = m.get("cond", np.nan); r["err_meas"] = m.get("err", np.nan)
        rows.append(r)
    json.dump(rows, open("phase_algebra/outputs/encoding_coherence.json", "w"), indent=1)
    real = rows[0]; sh = [r for r in rows[1:] if np.isfinite(r["err_meas"])]
    def rank_(v): o = np.argsort(v); rr = np.empty(len(v)); rr[o] = np.arange(len(v)); return rr
    ym = np.log(np.array([r["cond_meas"] for r in sh])); ye = np.log(np.array([r["err_meas"] for r in sh]))
    print("符号化コヒーレンス不変量 (ヌル n=%d):" % len(sh))
    print(f"{'不変量':16s} {'実配線':>8s} {'ヌル中央値':>10s} {'ρ→実測cond':>11s} {'ρ→実測誤差':>11s}  実配線の順位")
    for key, better in [("cond_soft", "low"), ("coh_mean", "high"), ("coh_min", "high"), ("roll_coh_mean", "high"), ("roll_coh_min", "high")]:
        x = np.array([r[key] for r in sh]); xl = np.log(x + 1e-9) if key == "cond_soft" else x
        rs_m = np.corrcoef(rank_(xl), rank_(ym))[0, 1]; rs_e = np.corrcoef(rank_(xl), rank_(ye))[0, 1]
        nb = int((x < real[key]).sum()) if better == "low" else int((x > real[key]).sum())
        print(f"{key:16s} {real[key]:8.3f} {np.median(x):10.3f} {rs_m:+11.3f} {rs_e:+11.3f}  ヌル{len(sh)}個中 {nb} 個が実配線より良い")


if __name__ == "__main__":
    main()
