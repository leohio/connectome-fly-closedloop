# phase_algebra

位相制御 + 線形代数による、ハエ飛行制御系の簡略化と解析。計画は [STRATEGY.md](STRATEGY.md)。

実行はリポジトリ直下から (外部データを相対パスで読むため):

```bash
../fly-body-sim/venv/bin/python phase_algebra/ident_plant.py      # P1: 機体 F, G の同定
```

| ファイル | 内容 |
|---|---|
| `ident_plant.py` | P1。ホバー動作点で微小摂動を与え、1羽ばたきのストロボ写像 F (9×9), G (9×5) を最小二乗で同定。固有値と線形予測の検証つき |
| `outputs/plant_FG.npz` | 同定結果 (F, G, c, 参照状態) |
