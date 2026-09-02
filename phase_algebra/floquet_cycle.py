#!/usr/bin/env python
"""P2 精密化: リミットサイクル周りの Floquet 乗数 (ポアンカレ写像の有限差分)。

ホバーは固定点ではなく ~11 Hz・振幅 ~0.1 rad のリミットサイクルなので、安定性の
正しい対象はサイクル周りの Floquet 乗数である。閉ループ (制御器 + 感覚模型 + 遅延 d) を
静定させ、羽ばたき平均ピッチ誤差 e_b1 の上向き平均交差をポアンカレ断面とする。
断面上のスナップショット (MjData + 制御器・感覚の内部状態) から、身体状態 9 方向に
小摂動 (+h) を与えて次の断面交差まで走らせ、断面上の偏差を h で割ってモノドロミー行列
M (9×9、身体部分空間への制限) を得る。固有値 μ が Floquet 乗数 (1 サイクルあたり)。
"""
import os, sys, json
import numpy as np
import mujoco
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import phase_algebra.ident_plant as IP
from phase_algebra.closed_loop import load_K, build_T, analyse, GS, A_S


class Loop:
    """閉ループ 1 羽ばたきを進める。内部状態を明示的に持つ (snapshot/restore 可能)"""
    def __init__(self, K, b, d):
        self.K, self.b, self.d = K, b, d

    def init(self, snap):
        self.d_ = IP.restore(snap)
        self.u_cmd = snap["u"].copy(); self.u = snap["u"].copy()
        self.buf = np.zeros((IP.STEPS, 3)); self.bi = 0; self.ssum = np.zeros(3)
        self.om_f = np.zeros(3); self.hist = [np.zeros(3) for _ in range(self.d + 1)]
        self.t = self.d_.time

    def snapshot(self):
        return dict(qpos=self.d_.qpos.copy(), qvel=self.d_.qvel.copy(), act=self.d_.act.copy(), time=self.d_.time,
                    u_cmd=self.u_cmd.copy(), u=self.u.copy(), buf=self.buf.copy(), bi=self.bi, ssum=self.ssum.copy(),
                    om_f=self.om_f.copy(), hist=[h.copy() for h in self.hist], t=self.t)

    def restore(self, s):
        self.d_ = IP.restore(s); self.u_cmd = s["u_cmd"].copy(); self.u = s["u"].copy()
        self.buf = s["buf"].copy(); self.bi = s["bi"]; self.ssum = s["ssum"].copy()
        self.om_f = s["om_f"].copy(); self.hist = [h.copy() for h in s["hist"]]; self.t = s["t"]

    def beat(self):
        """1 羽ばたき進め、その羽ばたきの平均状態を返す"""
        d_ = self.d_; acc = np.zeros(9)
        # 指令更新 (羽ばたき頭)
        om_bar = self.ssum / IP.STEPS
        self.om_f = A_S * self.om_f + (1 - A_S) * GS * om_bar
        self.hist.insert(0, self.om_f.copy()); self.hist = self.hist[:self.d + 1]
        om_hat = self.hist[self.d]
        x = IP.state_of(d_)
        xin = np.array([(IP.Z_T - x[5]) / 5.0, -x[6] / 30.0, x[0], x[1], om_hat[0] / 20, om_hat[1] / 20, om_hat[2] / 20])
        self.u_cmd = np.clip(IP.U_TRIM + np.tanh(self.K @ xin + self.b) * 0.35, -0.55, 0.55)
        for k in range(IP.STEPS):
            self.ssum += d_.qvel[3:6] - self.buf[self.bi]; self.buf[self.bi] = d_.qvel[3:6].copy(); self.bi = (self.bi + 1) % IP.STEPS
            self.u += IP.DTP * (self.u_cmd - self.u) / IP.TAU_TW
            IP.wing_ctrl(d_, self.u, self.t); mujoco.mj_step(IP.m, d_); self.t += IP.DTP
            acc += IP.state_of(d_)
        return acc / IP.STEPS


def run_to_section(loop, mean_eb1, max_beats=60, skip=3):
    """次の断面 (e_b1 平均の上向き交差) まで進め、(交差時の平均状態, 経過羽ばたき数) を返す"""
    prev = None
    for k in range(max_beats):
        xb = loop.beat()
        if k >= skip and prev is not None and prev[1] < mean_eb1 <= xb[1]:
            return xb, k + 1
        prev = xb
    return None, max_beats


def floquet(K, b, d, snap0, h=None):
    loop = Loop(K, b, d); loop.init(snap0)
    xs = [loop.beat() for _ in range(300)]                         # 1.5 s 静定
    if not np.isfinite(loop.d_.qpos[2]) or loop.d_.qpos[2] < 0.5: return None
    xs = np.array(xs[100:]); mean_eb1 = xs[:, 1].mean(); amp = xs[:, 1].std()
    # 断面へ
    xb, _ = run_to_section(loop, mean_eb1, skip=0)
    if xb is None: return None
    s0 = loop.snapshot()
    base, P = run_to_section(loop, mean_eb1)
    if base is None: return None
    if h is None: h = np.array([0.01, 0.01, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5])
    M = np.zeros((9, 9)); periods = []
    for i in range(9):
        loop.restore(s0)
        dx = np.zeros(9); dx[i] = h[i]
        IP.perturb(loop.d_, dx)
        xp, Pi = run_to_section(loop, mean_eb1)
        if xp is None: return None
        M[:, i] = (xp - base) / h[i]; periods.append(Pi)
    mu = np.linalg.eigvals(M)
    return dict(period_beats=P, period_ms=P * IP.PER * 1000, amp_eb1=amp, mu=mu, periods=periods)


if __name__ == "__main__":
    print("参照静定...", flush=True)
    snap0, _ = IP.settle(1.0)
    K, b = load_K("outputs/bioflight_best.npy")
    out = {}
    for d in (0, 1, 2):
        r = floquet(K, b, d, snap0)
        if r is None:
            print(f"d={d}: サイクルに到達できず (墜落)", flush=True); continue
        mu = sorted(r["mu"], key=lambda z: -abs(z)); P = r["period_beats"]
        per_beat = [abs(m) ** (1.0 / P) for m in mu[:4]]
        _, lam_fp, per_fp, _ = analyse(build_T(K, b, d, False, "circuit"))
        print(f"d={d}: サイクル周期 {P} 羽ばたき ({r['period_ms']:.0f} ms, 振幅 e_b1 std {r['amp_eb1']:.3f} rad)", flush=True)
        print(f"   Floquet乗数 |μ| (1サイクル) 上位4: " + ", ".join(f"{abs(m):.3f}" for m in mu[:4]))
        print(f"   → 1羽ばたきあたり |μ|^(1/P): " + ", ".join(f"{v:.4f}" for v in per_beat) + f"   (固定点線形化の速いモード予測 {lam_fp:.4f}, 周期 {per_fp:.0f} ms)")
        out[d] = dict(period_beats=int(P), amp=float(r["amp_eb1"]), mu_abs=[float(abs(m)) for m in mu], per_beat=[float(v) for v in per_beat], lam_fixed=float(lam_fp))
    json.dump(out, open("phase_algebra/outputs/floquet_cycle.json", "w"), indent=1)
    print("保存: phase_algebra/outputs/floquet_cycle.json")
