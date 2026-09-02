"""P4b: 身体を切り離した腹髄 (MANC LIF) の脚 MN 集団リズム

問い: MDN 緊張性駆動 (+ 一定の感覚入力) だけで、脚運動ニューロン群に
  (a) 2〜20 Hz の周期性 (CPG リズム)、(b) 脚間の一貫した位相関係、
  (c) 屈筋/伸筋の逆相交代 が出るか。
P4 で歩容が結合振動子として不成立だった原因が「腹髄にリズムが無い」のか
「身体との結合で壊れる」のかを切り分ける。

使い方: ../fly-body-sim/venv/bin/python phase_algebra/vnc_rhythm.py <T秒> <MDN Hz> <感覚 Hz> <tag>
"""
import sys, json, os
import numpy as np
from brian2 import Hz, ms as _ms, prefs
prefs.codegen.target = "numpy"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vnc_model, walk_base as WB, walk_drive as WD, walk_reflex as WR
import integrate_measured as IMm

T = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
MDN = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
SENS = float(sys.argv[3]) if len(sys.argv) > 3 else WR.R_TONIC
TAG = sys.argv[4] if len(sys.argv) > 4 else f"m{int(MDN)}_s{int(SENS)}"
BIN = 0.002   # 2 ms

W = WB.env()
sn_by_leg = WR.sensory_pools()
sn_all = [b for leg in sn_by_leg.values() for b in leg]
mp = IMm.build_measured_pools()
excit = {int(r.bodyid): float(IMm.E_MAX ** (1.0 - r.pct)) for _, r in mp.iterrows()}
tonic = {int(r.bodyid): float(IMm.ITN_MAX * max(0.0, (IMm.TONIC_CUTOFF - r.pct) / IMm.TONIC_CUTOFF))
         for _, r in mp.iterrows()}
net, mon, ids, idx, pg = vnc_model.make_network(
    vnc_model.MDN_BODYIDS, r_stim_hz=MDN, sensory_bodyids=sn_all,
    excitability=excit, tonic_mv=tonic)
pg.rates = SENS * Hz
pools = WD.flybody_pools(idx, W)
# 脚 × 方向 (dr=+1 屈曲 / −1 伸展) → ニューロン index 集合
groups = {}
for (act, dr), lst in pools.items():
    seg, sd = act.split("_")[1], act.split("_")[2]
    leg = [k for k, v in WR.LEG2SUF.items() if v == (seg, sd)][0]
    groups.setdefault((leg, "flex" if dr > 0 else "ext"), set()).update(ni for ni, *_ in lst)
groups = {k: np.array(sorted(v)) for k, v in groups.items()}
print("MN 群サイズ:", {f"{l}_{d}": len(v) for (l, d), v in groups.items()}, flush=True)

net.run(T * 1000 * _ms, report=None)
t = np.array(mon.t / _ms) / 1000.0
i = np.array(mon.i)
nb = int(T / BIN)
edges = np.arange(nb + 1) * BIN
rates = {}
for (leg, d), g in groups.items():
    sel = np.isin(i, g)
    h, _ = np.histogram(t[sel], bins=edges)
    rates[f"{leg}_{d}"] = h / BIN / max(len(g), 1)   # Hz / neuron
os.makedirs("phase_algebra/outputs", exist_ok=True)
np.savez(f"phase_algebra/outputs/vnc_rhythm_{TAG}.npz", bin=BIN, T=T, mdn=MDN, sens=SENS,
         spike_t=t, spike_i=i, **{k: v for k, v in rates.items()})

# ---- 解析: 0.5 s 以降を使う ----
k0 = int(0.5 / BIN)
def smooth(x, w=int(0.02 / BIN)):
    return np.convolve(x, np.ones(w) / w, "same")
res = {"T": T, "mdn": MDN, "sens": SENS, "legs": {}}
fs = 1.0 / BIN
band = (2.0, 20.0)
sig = {}
for k, r in rates.items():
    x = smooth(r[k0:]); x = x - x.mean()
    n = len(x)
    F = np.fft.rfft(x * np.hanning(n)); f = np.fft.rfftfreq(n, BIN); P = np.abs(F) ** 2
    m = (f >= band[0]) & (f <= band[1])
    fpk = float(f[m][np.argmax(P[m])]); ppk = float(P[m].max())
    # ピーク鋭さ: 帯域内ピーク / 帯域内中央値
    sharp = float(ppk / (np.median(P[m]) + 1e-12))
    res["legs"][k] = dict(mean_hz=float(r[k0:].mean()), peak_f=fpk, sharp=sharp,
                          cv=float(r[k0:].std() / (r[k0:].mean() + 1e-9)))
    # 帯域通過 Hilbert 位相
    Fb = F.copy(); Fb[~m] = 0
    xa = np.fft.irfft(Fb, n) ; xh = np.fft.irfft(Fb * (-1j) * np.sign(f), n)
    sig[k] = xa + 1j * xh
keys = sorted(sig.keys())
ph = {k: np.angle(sig[k]) for k in keys}
Z = np.zeros((len(keys), len(keys)))
for a, ka in enumerate(keys):
    for b, kb in enumerate(keys):
        Z[a, b] = abs(np.mean(np.exp(1j * (ph[ka] - ph[kb]))))
res["coh_keys"] = keys; res["coh"] = Z.round(3).tolist()
# 屈筋-伸筋 交代指数 (同脚): cos(Δφ) の平均 (−1 = 逆相)
alt = {}
for leg in WR.LEGS:
    a, b = f"{leg}_flex", f"{leg}_ext"
    if a in ph and b in ph:
        alt[leg] = float(np.mean(np.cos(ph[a] - ph[b])))
res["flex_ext_cos"] = alt
# 脚間 (屈筋同士) コヒーレンス
fl = [f"{l}_flex" for l in WR.LEGS if f"{l}_flex" in ph]
Zf = np.array([[abs(np.mean(np.exp(1j * (ph[a] - ph[b])))) for b in fl] for a in fl])
res["flex_legs"] = fl; res["flex_coh"] = Zf.round(3).tolist()
ev = np.linalg.eigvalsh(Zf + 0j); res["flex_coh_lead_eig"] = float(ev[-1].real)
json.dump(res, open(f"phase_algebra/outputs/vnc_rhythm_{TAG}.json", "w"), indent=1, ensure_ascii=False)

print(f"\n=== MDN {MDN:.0f} Hz, 感覚 {SENS:.0f} Hz, {T:.1f} s ===")
print("群       平均Hz/MN  帯域ピークf  鋭さ   CV")
for k in keys:
    r = res["legs"][k]
    print(f"{k:8s} {r['mean_hz']:8.2f} {r['peak_f']:9.2f} {r['sharp']:7.1f} {r['cv']:5.2f}")
print("屈筋-伸筋 cos(Δφ) (−1=逆相交代):", {k: round(v, 2) for k, v in alt.items()})
print("脚間 (屈筋) |Z|:")
print("      " + " ".join(f"{a[:2]:>5s}" for a in fl))
for a, row in zip(fl, Zf):
    print(f"{a[:2]:>5s} " + " ".join(f"{v:5.2f}" for v in row))
print(f"主固有値 {res['flex_coh_lead_eig']:.2f} / {len(fl)}")
