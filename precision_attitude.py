#!/usr/bin/env python
"""統合33: 飽和しない行動指標で、MANC配線の感覚精度を検定する。

従来の3秒ホバリングは実配線も旧部分ヌルも上限まで生存し、復号誤差
(実配線0.113、部分ヌル平均0.191) の差を行動から読めなかった。統合33では
補完電気経路も置換する完全ヌルを用いる。
ここでは飛行中の胸部へ決定論的な多周波トルクを加え、次を連続量で測る。

  - 姿勢誤差 RMS / 95百分位
  - 低周波角速度 RMS
  - 高度誤差
  - 生存時間 (安全性確認用。主指標にはしない)

感覚以外は統合31と同一である。神経筋制約も維持する:
1羽ばたき1回の指令更新、筋単収縮 tau=4.25 ms、全帯域ジャイロ不使用。
外力は「乱流・突風に対する姿勢外乱」の実験操作であり、制御入力ではない。
"""
import json
import os
import sys
from functools import lru_cache
from multiprocessing import Pool

import mujoco
import numpy as np
from brian2 import ms as _ms, prefs

prefs.codegen.target = "numpy"

import connectome_bioflight as CB
import connectome_fastloop as CF
import locked_circuit as LC
import openloop_hover as OH
import phase_reflex as PR


T_DEFAULT = 2.5
BURN_IN = 0.35
DEFAULT_AMP = 2.0e-3
FREQS = np.array([[3.0, 7.0, 11.0],
                  [4.0, 8.0, 13.0],
                  [5.0, 9.0, 14.0]])
WEIGHTS = np.array([1.0, 0.55, 0.30])


@lru_cache(maxsize=None)
def _disturbance_phases(seed):
    """試行中不変の位相をseedごとに1回だけ生成する。"""
    return np.random.default_rng(9000 + seed).uniform(0, 2 * np.pi, (3, 3))


def disturbance(t, amp, seed=0):
    """再現可能な3軸多周波トルク [world frame]。

    seed は周波数を変えず位相だけを変えるので、全配線条件は同じ帯域を
    異なる実現値で経験する。各軸のRMSがおおむね amp になるよう正規化する。
    """
    if t < BURN_IN or amp == 0:
        return np.zeros(3)
    phases = _disturbance_phases(seed)
    tt = t - BURN_IN
    x = np.sum(WEIGHTS[None, :] *
               np.sin(2 * np.pi * FREQS * tt + phases), axis=1)
    return amp * x / np.sqrt(np.sum(WEIGHTS ** 2) / 2.0)


def fly(src="true", dec=None, T=T_DEFAULT, pert_seed=0, amp=DEFAULT_AMP,
        shuffle=None, seed=0, trace=False):
    """統合31の飛行器を変更せず、外乱とgraded指標だけを追加する。"""
    E = OH.env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    thorax = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thorax")
    use_net = src in ("circuit", "shuffle")
    if use_net:
        Sg, names, phi0 = dec
        kw = dict(LC.SETUP_KW)
        if shuffle:
            kw["shuffle"] = shuffle
            kw["shuffle_seed"] = seed
        net, mon, pg, pref, side, st_idx, n = PR.setup(**kw)
        idx_of = {j: st_idx[j] for j in names if j in st_idx}
        zc = {j: 0j for j in names}
        prev = 0
        shift_prev = np.zeros(len(pref))

    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)

    dtp = m.opt.timestep
    P0, U_TRIM = CB.P0, CB.U_TRIM
    kk = max(P0["sharp"], 1e-3)
    R = np.zeros(9)
    per = 1.0 / P0["freq"]
    nbox = max(int(per / CF.DT_N), 1)
    ombuf = np.zeros((nbox, 3))
    ombi = 0
    omsum = np.zeros(3)
    u_cmd = np.array(U_TRIM, float)
    u = np.array(U_TRIM, float)
    om_est = np.zeros(3)
    t_next = 0.0
    n_sub = max(int(round(CF.DT_N / dtp)), 1)

    tilt, omega, zerr, times = [], [], [], []
    dec_err_clip, dec_ref_clip = [], []
    dec_err_full, dec_ref_full = [], []
    alive = 0
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        omsum += d.qvel[3:6] - ombuf[ombi]
        ombuf[ombi] = d.qvel[3:6].copy()
        ombi = (ombi + 1) % nbox
        o_slow = omsum / nbox

        if use_net:
            o = np.clip(o_slow, -12, 12)
            shift = PR.C_PHASE * (side * o[0] + o[1]
                                  + side * np.cos(2 * np.pi * pref) * o[2])
            pg.v = pg.v - (shift - shift_prev)
            shift_prev = shift
            net.run(CF.DT_N * 1000 * _ms, namespace={})
            ph_w = (tn * P0["freq"]) % 1.0
            nsp = mon.num_spikes
            if nsp > prev:
                ev = np.exp(2j * np.pi * ph_w)
                for iN in np.array(mon.i[prev:nsp]):
                    for j, im in idx_of.items():
                        if int(iN) == im:
                            zc[j] += ev
                prev = nsp
            dphi = np.zeros(len(names))
            ok = True
            for q, j in enumerate(names):
                zc[j] -= CF.DT_N * zc[j] / LC.TAU_P
                if abs(zc[j]) < 1e-3:
                    ok = False
                    break
                dd = np.angle(zc[j]) / (2 * np.pi) - phi0[q]
                dphi[q] = (dd + 0.5) % 1.0 - 0.5
            if ok and tn > 0.12:
                om_est = dphi @ Sg

        if tn >= t_next:
            t_next += per
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zcv = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zcv, ZT_W)
            # Brian2 の膜電位も ``v`` という名前なので、局所変数を v と呼ぶと
            # net.run() が毎ステップ名前衝突警告を出す。数値には無関係だが、
            # ログ肥大化と大幅な低速化を招くため明示的に避ける。
            vel = d.qvel[:3]
            if src == "true":
                ow = o_slow
            elif src == "zero":
                ow = np.zeros(3)
            else:
                ow = om_est
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -vel[2] / 30.0,
                          e_b[0], e_b[1],
                          ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u_cmd = np.clip(U_TRIM +
                            np.tanh(CB.K_POL @ x + CB.B_POL) * 0.35,
                            -0.55, 0.55)

        for _ in range(n_sub):
            t = d.time
            d.xfrc_applied[thorax, :] = 0.0
            d.xfrc_applied[thorax, 3:6] = disturbance(t, amp, pert_seed)
            u += dtp * (u_cmd - u) / CB.TAU_TW
            amp_w = np.clip(1.0 + u[0], 0.5, 1.6)
            env0 = min(t / 0.03, 1.0)
            ph2 = 2 * np.pi * P0["freq"] * t
            s = np.sin(ph2)
            rot = np.tanh(kk * np.cos(ph2 + P0["phase"])) / np.tanh(kk)
            e = env0 * amp_w
            d.ctrl[:] = 0
            d.ctrl[aid["wing_yaw_left"]] = e * (P0["yaw_amp"] * s + u[1] + u[2])
            d.ctrl[aid["wing_yaw_right"]] = e * (P0["yaw_amp"] * s + u[1] - u[2])
            d.ctrl[aid["wing_pitch_left"]] = e * (-P0["pitch_amp"] * rot
                                                   + P0["pitch_bias"] + u[3] + u[4])
            d.ctrl[aid["wing_pitch_right"]] = e * (-P0["pitch_amp"] * rot
                                                    + P0["pitch_bias"] + u[3] - u[4])
            d.ctrl[aid["wing_roll_left"]] = e * P0["roll_amp"] * np.sin(2 * ph2)
            d.ctrl[aid["wing_roll_right"]] = e * P0["roll_amp"] * np.sin(2 * ph2)
            mujoco.mj_step(m, d)

        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5 or d.qpos[2] > 40:
            break
        if tn >= BURN_IN:
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zcv = np.array([R[2], R[5], R[8]])
            eb = R.reshape(3, 3).T @ np.cross(zcv, ZT_W)
            tilt.append(float(np.linalg.norm(eb[:2])))
            omega.append(float(np.linalg.norm(o_slow)))
            zerr.append(float(abs(d.qpos[2] - 12.0)))
            times.append(tn)
            if use_net:
                # 回路が実際に受け取る飽和後の目標と、身体の全角速度の両方に
                # 対するオンライン誤差を残す。位相遅れも行動上の誤差なので含む。
                oc = np.clip(o_slow, -12, 12)
                dec_err_clip.append(float(np.sum((om_est - oc) ** 2)))
                dec_ref_clip.append(float(np.sum(oc ** 2)))
                dec_err_full.append(float(np.sum((om_est - o_slow) ** 2)))
                dec_ref_full.append(float(np.sum(o_slow ** 2)))
        alive = kn + 1

    tv = np.asarray(tilt)
    ov = np.asarray(omega)
    zv = np.asarray(zerr)
    nrmse_clip = (float(np.sqrt(np.sum(dec_err_clip) /
                                max(np.sum(dec_ref_clip), 1e-12)))
                  if dec_err_clip else None)
    nrmse_full = (float(np.sqrt(np.sum(dec_err_full) /
                                max(np.sum(dec_ref_full), 1e-12)))
                  if dec_err_full else None)
    # 早期墜落後が観測RMSから消える打切り偏りを除く。評価開始BURN_INから
    # Tまでを固定窓とし、墜落後は最大傾斜誤差1.0 (水平から90度) で埋める。
    # 任意の罰則係数ではなく、姿勢誤差そのものの固定時間RMSである。
    horizon = max(T - BURN_IN, CF.DT_N)
    observed = min(len(tv) * CF.DT_N, horizon)
    missing = max(horizon - observed, 0.0)
    tilt_fixed = float(np.sqrt((np.sum(tv ** 2) * CF.DT_N + missing) /
                               horizon))
    result = dict(src=src, seed=int(seed), pert_seed=int(pert_seed),
                  amp=float(amp), survival=float(alive * CF.DT_N),
                  tilt_rms=float(np.sqrt(np.mean(tv ** 2))) if len(tv) else 1.0,
                  tilt_fixed_rms=tilt_fixed,
                  tilt_p95=float(np.quantile(tv, 0.95)) if len(tv) else 1.0,
                  omega_rms=float(np.sqrt(np.mean(ov ** 2))) if len(ov) else 100.0,
                  z_mae=float(np.mean(zv)) if len(zv) else 20.0,
                  decode_nrmse_clip=nrmse_clip,
                  decode_nrmse_full=nrmse_full)
    if trace:
        result["time"] = np.asarray(times)
        result["tilt"] = tv
        result["omega"] = ov
        result["zerr"] = zv
    return result


def _probe_job(args):
    src, amp, pert = args
    return fly(src=src, amp=amp, pert_seed=pert)


def probe():
    """完全感覚/感覚喪失で、情報量のある外乱強度を選ぶ。"""
    amps = [0.0, 2e-5, 5e-5, 1e-4, 2e-4, 4e-4]
    jobs = [(src, amp, seed) for amp in amps
            for src in ("true", "zero") for seed in range(2)]
    with Pool(min(12, len(jobs))) as pool:
        out = pool.map(_probe_job, jobs)
    for amp in amps:
        print(f"amp={amp:.1e}", flush=True)
        for src in ("true", "zero"):
            v = [r for r in out if r["amp"] == amp and r["src"] == src]
            print(f"  {src:4s}: 生存{np.mean([r['survival'] for r in v]):.2f}s "
                  f"tilt_rms={np.mean([r['tilt_rms'] for r in v]):.3f} "
                  f"omega_rms={np.mean([r['omega_rms'] for r in v]):.2f} "
                  f"z_mae={np.mean([r['z_mae'] for r in v]):.2f}", flush=True)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/precision_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print("DONE", flush=True)


def load_decoder(src, seed):
    """配線ごとの較正結果を読み、無い場合だけ直接MANC試行で作る。"""
    sh = "all" if src == "shuffle" else None
    # fullnull は化学配線だけでなく、補完した電気シナプスの対応もshuffleする。
    # 旧 precision_dec_shuffle_* はpartial-nullの再現用に残し、混用しない。
    tag = "real" if src == "circuit" else f"fullnull_shuffle_{seed}"
    cache = f"outputs/precision_dec_{tag}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        dec = (z["Sg"], [str(x) for x in z["names"]], z["phi0"])
        return dec, sh, "cache"
    dec = CB.calibrate(shuffle=sh, seed=seed)
    np.savez(cache, Sg=dec[0], names=np.asarray(dec[1]), phi0=dec[2])
    return dec, sh, "new"


def one(argv):
    src = argv[0] if argv else "true"
    seed = int(argv[1]) if len(argv) > 1 else 0      # 配線seed
    amp = float(argv[2]) if len(argv) > 2 else DEFAULT_AMP
    pert = int(argv[3]) if len(argv) > 3 else seed  # 外乱seed (明示推奨)
    duration = float(argv[4]) if len(argv) > 4 else T_DEFAULT
    dec, sh = None, None
    if src in ("circuit", "shuffle"):
        dec, sh, how = load_decoder(src, seed)
        print(f"較正{'キャッシュ読込' if how == 'cache' else '完了・保存'}: "
              f"{len(dec[1])}筋", flush=True)
    r = fly(src=src, dec=dec, T=duration, amp=amp, pert_seed=pert,
            shuffle=sh, seed=seed)
    print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)


MATCHED = [("true", 0, "true"),
           ("circuit", 0, "real"),
           # 旧partial-nullでのbest/median等の順位は、電気経路も交換する
           # full-nullでは意味が変わるため、後付けの品質名を使わずseedで呼ぶ。
           ("shuffle", 6, "shuffle_6"),
           ("shuffle", 21, "shuffle_21"),
           ("shuffle", 10, "shuffle_10"),
           ("shuffle", 38, "shuffle_38"),
           ("zero", 0, "zero")]


def _matched_job(arg):
    src, wiring_seed, label, pert, amp, duration = arg
    dec, sh = None, None
    if src in ("circuit", "shuffle"):
        dec, sh, _ = load_decoder(src, wiring_seed)
    r = fly(src=src, dec=dec, T=duration, amp=amp, pert_seed=pert,
            shuffle=sh, seed=wiring_seed)
    r["label"] = label
    r["wiring_seed"] = wiring_seed
    return r


def matched(argv):
    """同一外乱位相で条件を対応させる本試験。"""
    n_pert = int(argv[0]) if argv else 3
    amp = float(argv[1]) if len(argv) > 1 else DEFAULT_AMP
    duration = float(argv[2]) if len(argv) > 2 else 2.0
    jobs = [(src, ws, label, pert, amp, duration)
            for pert in range(n_pert) for src, ws, label in MATCHED]
    with Pool(min(12, len(jobs))) as pool:
        out = pool.map(_matched_job, jobs)
    for label in [x[2] for x in MATCHED]:
        v = [r for r in out if r["label"] == label]
        dn = [r["decode_nrmse_clip"] for r in v
              if r["decode_nrmse_clip"] is not None]
        ds = f" decode={np.mean(dn):.2f}" if dn else ""
        print(f"{label:15s}: 生存{np.mean([r['survival'] for r in v]):.2f}s "
              f"tilt={np.mean([r['tilt_rms'] for r in v]):.3f} "
              f"p95={np.mean([r['tilt_p95'] for r in v]):.3f} "
              f"omega={np.mean([r['omega_rms'] for r in v]):.2f} "
              f"z={np.mean([r['z_mae'] for r in v]):.2f}{ds}", flush=True)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/precision_matched.json", "w") as f:
        json.dump(out, f, indent=2)
    print("保存 outputs/precision_matched.json", flush=True)
    print("DONE", flush=True)


def _fixed_job(arg):
    src, wiring_seed, label, pert, amp, duration = arg
    dec, sh = None, None
    if src in ("circuit", "shuffle"):
        # downstream decoder/基準位相は実配線で学んだものから一切変えない。
        dec, _, _ = load_decoder("circuit", 0)
        sh = "all" if src == "shuffle" else None
    r = fly(src=src, dec=dec, T=duration, amp=amp, pert_seed=pert,
            shuffle=sh, seed=wiring_seed)
    r["label"] = label
    r["wiring_seed"] = wiring_seed
    return r


def fixed(argv):
    """実配線で学んだ下流復号器を固定し、配線だけを交換する転用試験。

    matched は各配線に同じ再学習機会を与える問い、fixed は発達・学習済みの
    読出しに対し、その上流配線を入れ替えても機能するかという別の問いである。
    """
    n_pert = int(argv[0]) if argv else 3
    amp = float(argv[1]) if len(argv) > 1 else DEFAULT_AMP
    duration = float(argv[2]) if len(argv) > 2 else 2.0
    jobs = [(src, ws, label, pert, amp, duration)
            for pert in range(n_pert) for src, ws, label in MATCHED]
    with Pool(min(12, len(jobs))) as pool:
        out = pool.map(_fixed_job, jobs)
    for label in [x[2] for x in MATCHED]:
        vals = [r for r in out if r["label"] == label]
        dn = [r["decode_nrmse_clip"] for r in vals
              if r["decode_nrmse_clip"] is not None]
        ds = f" decode={np.mean(dn):.2f}" if dn else ""
        print(f"{label:15s}: 生存{np.mean([r['survival'] for r in vals]):.2f}s "
              f"tilt={np.mean([r['tilt_rms'] for r in vals]):.3f} "
              f"p95={np.mean([r['tilt_p95'] for r in vals]):.3f} "
              f"omega={np.mean([r['omega_rms'] for r in vals]):.2f} "
              f"z={np.mean([r['z_mae'] for r in vals]):.2f}{ds}", flush=True)
    with open("outputs/precision_fixed.json", "w") as f:
        json.dump(out, f, indent=2)
    print("保存 outputs/precision_fixed.json", flush=True)
    print("DONE", flush=True)


def _fixed_population_job(arg):
    wiring_seed, pert, amp, duration = arg
    dec, _, _ = load_decoder("circuit", 0)
    r = fly(src="shuffle", dec=dec, T=duration, amp=amp, pert_seed=pert,
            shuffle="all", seed=wiring_seed)
    r["label"] = f"shuffle_{wiring_seed}"
    r["wiring_seed"] = wiring_seed
    return r


def fixed_population(argv):
    """固定読出しをfull-null多数配線へ転用する配線標本の検定。

    配線seedが統計単位。外乱seedは全配線で同じ値に事前固定し、同じ身体入力を
    与える。実配線以上に飛べるfull-null数から片側経験的p値を求める。
    """
    n_shuffle = int(argv[0]) if argv else 40
    pert = int(argv[1]) if len(argv) > 1 else 0
    amp = float(argv[2]) if len(argv) > 2 else DEFAULT_AMP
    duration = float(argv[3]) if len(argv) > 3 else 2.0
    dec, _, _ = load_decoder("circuit", 0)
    real = fly(src="circuit", dec=dec, T=duration, amp=amp,
               pert_seed=pert, seed=0)
    jobs = [(seed, pert, amp, duration) for seed in range(n_shuffle)]
    with Pool(min(12, len(jobs))) as pool:
        out = pool.map(_fixed_population_job, jobs)
    srv = np.array([r["survival"] for r in out])
    tilt_v = np.array([r["tilt_rms"] for r in out])
    fixed_v = np.array([r["tilt_fixed_rms"] for r in out])
    n_ge = int(np.sum(srv >= real["survival"] - 1e-9))
    p_emp = (n_ge + 1) / (n_shuffle + 1)
    n_fixed = int(np.sum(fixed_v <= real["tilt_fixed_rms"] + 1e-12))
    p_fixed = (n_fixed + 1) / (n_shuffle + 1)
    print(f"実配線: 生存{real['survival']:.2f}s tilt={real['tilt_rms']:.3f} "
          f"decode={real['decode_nrmse_clip']:.2f}", flush=True)
    print(f"full-null {n_shuffle}配線: 生存{srv.mean():.2f}±{srv.std():.2f}s "
          f"範囲[{srv.min():.2f},{srv.max():.2f}] "
          f"tilt={tilt_v.mean():.3f}±{tilt_v.std():.3f}", flush=True)
    print(f"実配線以上の生存: {n_ge}/{n_shuffle}; "
          f"片側経験的p={p_emp:.4f}", flush=True)
    print(f"固定時間姿勢RMS: 実={real['tilt_fixed_rms']:.3f}, "
          f"full-null={fixed_v.mean():.3f}±{fixed_v.std():.3f}; "
          f"実以下={n_fixed}/{n_shuffle}; p={p_fixed:.4f}", flush=True)
    payload = dict(real=real, shuffles=out, n_ge=n_ge, p_emp=p_emp,
                   n_fixed_le=n_fixed, p_fixed=p_fixed,
                   n_shuffle=n_shuffle, pert_seed=pert, amp=amp,
                   duration=duration)
    path = f"outputs/precision_fixed_population_p{pert}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"保存 {path}", flush=True)
    print("DONE", flush=True)


def _matched_population_job(arg):
    """full-null配線に、その配線自身で較正し直した読出しを与える。"""
    wiring_seed, pert, amp, duration = arg
    try:
        dec, sh, _ = load_decoder("shuffle", wiring_seed)
    except Exception as exc:
        return dict(label=f"shuffle_{wiring_seed}", wiring_seed=wiring_seed,
                    failed=str(exc), survival=0.0, tilt_rms=1.0,
                    tilt_fixed_rms=1.0, tilt_p95=1.0, omega_rms=0.0,
                    z_mae=20.0, decode_nrmse_clip=None,
                    decode_nrmse_full=None)
    r = fly(src="shuffle", dec=dec, T=duration, amp=amp, pert_seed=pert,
            shuffle=sh, seed=wiring_seed)
    r["label"] = f"shuffle_{wiring_seed}"
    r["wiring_seed"] = wiring_seed
    return r


def matched_population(argv):
    """統合34: 各配線に自前の読出しを与えたときも実配線が優れるかの検定。

    fixed-pop は「実配線用の読出しを固定して上流を交換する」問いであり、
    読出しが不整合になる以上ヌルが劣るのは構成上ほぼ自明である。
    ここでは全配線に等しく再較正の機会を与え、それでも実配線が
    full-null分布の外に出るかを見る。統合33の matched (ヌル4構成) の母集団版。
    """
    n_shuffle = int(argv[0]) if argv else 40
    pert = int(argv[1]) if len(argv) > 1 else 2
    amp = float(argv[2]) if len(argv) > 2 else DEFAULT_AMP
    duration = float(argv[3]) if len(argv) > 3 else 2.0
    dec, _, _ = load_decoder("circuit", 0)
    real = fly(src="circuit", dec=dec, T=duration, amp=amp,
               pert_seed=pert, seed=0)
    jobs = [(seed, pert, amp, duration) for seed in range(n_shuffle)]
    with Pool(min(12, len(jobs))) as pool:
        out = pool.map(_matched_population_job, jobs)
    ok = [r for r in out if not r.get("failed")]
    n_fail = len(out) - len(ok)
    fixed_v = np.array([r["tilt_fixed_rms"] for r in ok])
    srv = np.array([r["survival"] for r in ok])
    n_le = int(np.sum(fixed_v <= real["tilt_fixed_rms"] + 1e-12))
    p_fixed = (n_le + 1) / (len(ok) + 1)
    n_ge = int(np.sum(srv >= real["survival"] - 1e-9))
    p_emp = (n_ge + 1) / (len(ok) + 1)
    print(f"各配線に自前の読出しを与えた場合 (外乱位相{pert}, n={len(ok)}"
          f"{'、較正失敗' + str(n_fail) if n_fail else ''})", flush=True)
    print(f"  実配線     : 固定時間姿勢RMS={real['tilt_fixed_rms']:.4f} "
          f"生存{real['survival']:.2f}s", flush=True)
    print(f"  full-null  : 中央値={np.median(fixed_v):.4f} "
          f"IQR={np.percentile(fixed_v, 25):.4f}-"
          f"{np.percentile(fixed_v, 75):.4f} "
          f"平均={fixed_v.mean():.4f}±{fixed_v.std():.4f}", flush=True)
    print(f"  実配線以下 : {n_le}/{len(ok)}  片側経験的p={p_fixed:.4f}",
          flush=True)
    print(f"  生存で見ると: 実以上={n_ge}/{len(ok)} p={p_emp:.4f}", flush=True)
    payload = dict(real=real, shuffles=out, n_le=n_le, p_fixed=p_fixed,
                   n_ge=n_ge, p_emp=p_emp, n_shuffle=len(ok),
                   n_failed=n_fail, pert_seed=pert, amp=amp,
                   duration=duration)
    path = f"outputs/precision_matched_population_p{pert}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"保存 {path}", flush=True)
    print("DONE", flush=True)


def _legacy_fixed_cost(q, duration):
    """tilt_fixed_rms追加前に保存した結果も同じ定義で読めるようにする。"""
    if "tilt_fixed_rms" in q:
        return float(q["tilt_fixed_rms"])
    horizon = max(duration - BURN_IN, CF.DT_N)
    observed = max(min(q["survival"], duration) - BURN_IN, 0.0)
    missing = max(horizon - observed, 0.0)
    return float(np.sqrt((q["tilt_rms"] ** 2 * observed + missing) / horizon))


def combine_population(argv):
    """複数外乱位相の40配線結果を配線seedごとに平均して最終検定する。"""
    perts = [int(x) for x in argv] if argv else [0, 1]
    runs = []
    for pert in perts:
        path = f"outputs/precision_fixed_population_p{pert}.json"
        with open(path) as f:
            runs.append(json.load(f))
    seeds = sorted(set(q["wiring_seed"] for q in runs[0]["shuffles"]))
    real_cost = float(np.mean([_legacy_fixed_cost(x["real"], x["duration"])
                               for x in runs]))
    rows = []
    for seed in seeds:
        vals = []
        for x in runs:
            q = next(q for q in x["shuffles"] if q["wiring_seed"] == seed)
            vals.append(_legacy_fixed_cost(q, x["duration"]))
        rows.append(dict(wiring_seed=seed, mean_fixed_rms=float(np.mean(vals)),
                         per_pert=vals))
    cv = np.array([r["mean_fixed_rms"] for r in rows])
    n_le = int(np.sum(cv <= real_cost + 1e-12))
    p = (n_le + 1) / (len(cv) + 1)
    print(f"外乱位相{perts}: 実配線 固定時間RMS={real_cost:.4f}", flush=True)
    print(f"full-null {len(cv)}配線={cv.mean():.4f}±{cv.std():.4f}; "
          f"実以下={n_le}/{len(cv)}; 片側経験的p={p:.4f}", flush=True)
    with open("outputs/precision_fixed_population_multi.json", "w") as f:
        json.dump(dict(perts=perts, real_fixed_rms=real_cost,
                       shuffles=rows, n_le=n_le, p_emp=p), f, indent=2)
    print("保存 outputs/precision_fixed_population_multi.json", flush=True)
    print("DONE", flush=True)


def _synapse_edges(net, prefix):
    """Networkから指定Synapsesの(pre, post)を安定した順序で取り出す。"""
    found = [obj for obj in net.objects
             if obj.__class__.__name__ == "Synapses"
             and obj.name.startswith(prefix)]
    if len(found) != 1:
        raise RuntimeError(f"{prefix}: Synapsesを一意に特定できない ({len(found)})")
    syn = found[0]
    return np.asarray(syn.i[:], dtype=int), np.asarray(syn.j[:], dtype=int)


def audit_null(argv):
    """full-nullが化学・補完電気の両経路で次数を保存することを監査する。"""
    seed = int(argv[0]) if argv else 0
    fresh = np.random.default_rng(9000 + seed).uniform(0, 2 * np.pi, (3, 3))
    cache_equivalent = bool(np.array_equal(_disturbance_phases(seed), fresh))
    if not cache_equivalent:
        raise AssertionError("disturbance phase cache changed the waveform")
    net0, *_ = PR.setup(**LC.SETUP_KW)
    kw = dict(LC.SETUP_KW, shuffle="all", shuffle_seed=seed)
    net1, *_ = PR.setup(**kw)
    out = {"shuffle_seed": seed,
           "disturbance_cache_equivalent": cache_equivalent}
    for label, prefix in (("chemical", "ssyn"),
                          ("modeled_electrical", "elsyn")):
        i0, j0 = _synapse_edges(net0, prefix)
        i1, j1 = _synapse_edges(net1, prefix)
        if len(i0) != len(i1):
            raise AssertionError(f"{label}: edge count changed")
        pre_preserved = bool(np.array_equal(i0, i1))
        # bincountの長さを共通化し、各targetへの入力本数が同じか調べる。
        npost = int(max(j0.max(initial=0), j1.max(initial=0)) + 1)
        indegree_preserved = bool(np.array_equal(
            np.bincount(j0, minlength=npost),
            np.bincount(j1, minlength=npost)))
        same = int(np.sum((i0 == i1) & (j0 == j1)))
        row = dict(n_edges=int(len(i0)), pre_and_outdegree_preserved=pre_preserved,
                   post_indegree_preserved=indegree_preserved,
                   same_ordered_pairs=same,
                   same_pair_fraction=float(same / max(len(i0), 1)))
        if not pre_preserved or not indegree_preserved:
            raise AssertionError(f"{label}: degree-preserving null invariant failed")
        out[label] = row
        print(f"{label}: edges={len(i0)} same={same} "
              f"({row['same_pair_fraction']:.3%}) pre/out保存={pre_preserved} "
              f"post/in保存={indegree_preserved}", flush=True)
    with open("outputs/full_null_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("保存 outputs/full_null_audit.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe()
    elif len(sys.argv) > 1 and sys.argv[1] == "matched":
        matched(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "fixed":
        fixed(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "fixed-pop":
        fixed_population(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "matched-pop":
        matched_population(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "combine-pop":
        combine_population(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "audit-null":
        audit_null(sys.argv[2:])
    else:
        one(sys.argv[1:])
