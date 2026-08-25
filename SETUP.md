# SETUP — 実行に必要な外部データと配置

このリポジトリは元のモノレポ `physical-based-llm` の `fly-integrated/` を
履歴ごと切り出したものである。**コードは兄弟ディレクトリを相対パスで参照する**ため、
そのまま実行するには以下の作業ツリーを再現する必要がある。

## 期待するディレクトリ構成

```
workspace/
├── connectome-fly-closedloop/        ← このリポジトリ
│   ├── neuron-properties.feather     ← 別途配置が必要 (16MB, 下記)
│   └── *.py
├── vnc-connectome/downloads/         ← MANC 腹髄コネクトーム
│   ├── traced-connections.csv        (72MB)
│   ├── traced-neurons.csv            (616KB)
│   ├── elife-96084-supp3-v1.csv      (翅MN ↔ 筋の対応表)
│   └── elife-96084-supp6-v1.csv      (脚MN の対応表)
├── fly-flight-sim/                   ← flybody (MuJoCo身体) と基準波形
│   ├── flybody/                      (237MB, Janelia flybody)
│   └── outputs/hover_policy.npz, hover_params.npz
├── fly-brain-sim/Drosophila_brain_model/  ← FlyWire 全脳 (404MB)
│   ├── coordinates.csv               (脳ニューロンの実座標)
│   ├── classification.csv, consolidated_cell_types.csv
│   └── Connectivity_783.parquet
└── fly-body-sim/venv/                ← Python環境
```

## 依存の内訳(どのスクリプトが何を読むか)

| 外部ファイル | 用途 | 主な利用元 |
|---|---|---|
| `traced-connections.csv` | MANC の実配線(化学シナプス) | `vnc_model.py` → 全て |
| `traced-neurons.csv` | ニューロンの型・側性 | `vnc_model.py` |
| `elife-96084-supp3-v1.csv` | 翅の運動ニューロンと筋の対応 | `phase_reflex.py`, `measure_S.py` |
| `elife-96084-supp6-v1.csv` | 脚の運動ニューロンの対応 | `vnc_model.leg_mn_table()` |
| `neuron-properties.feather` | ソーマ座標・クラス・サブクラス等の属性 | 11スクリプト(可視化・感覚型分離) |
| `hover_policy.npz` / `hover_params.npz` | 基準となる翅運動学と帰還則 | `phase_reflex._fly_env()` → 全飛行 |
| `flybody/` | MuJoCo の身体モデル | `optics.get_model()` |
| `Drosophila_brain_model/` | FlyWire 視覚回路と脳の実座標 | `brain_visual.py`, `video_sequence.py` |

## データの出所(すべて公開データ)

- **MANC v1.0**(腹髄コネクトーム): Janelia の公開リリース。
  `traced-connections.csv` / `traced-neurons.csv` と、eLife 96084 の補足表
- **FlyWire**(全脳コネクトーム): Drosophila brain model の公開配布物
- **flybody**: Janelia の MuJoCo ハエ身体モデル
- `neuron-properties.feather` は MANC のニューロン属性テーブルを feather 化したもの
  (サイズの都合でリポジトリには含めていない)

## Python環境

`brian2 2.6.0` / `mujoco 3.2.3` / `numpy 1.26.4` / `pandas` / `imageio` / `Pillow`。
`prefs.codegen.target = "numpy"` を全スクリプト冒頭で設定している
(Cython は並列実行時にコンパイルキャッシュが競合するため使わない)。

```bash
../fly-body-sim/venv/bin/python connectome_bioflight.py circuit 0
```

**作業ディレクトリは必ずこのリポジトリの直下**であること(相対パスで外部データを読むため)。

## 注意

- 実行ログは brian2 の警告で数百MBに肥大化するため `.gitignore` 済み
- `LOG.md` 中で参照しているコミットハッシュは**切り出し前のモノレポのもの**であり、
  このリポジトリのハッシュとは対応しない(履歴の内容自体は保持されている)
