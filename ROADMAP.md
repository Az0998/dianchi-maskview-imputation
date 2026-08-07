# Dianchi Mask-View 插补 → 预警 科研路线

## 论文核心卖点

**Mask-View augmentation**：针对自动站 **块状缺失**（>99% block-structured），用多模式掩码视图（point / block_time / sensor / station / mixed）+ 重建损失 + 跨视图一致性损失训练时空插补器。

## 优先级落地状态

| 模块 | 状态 | 入口 |
|------|------|------|
| ① Mask-View augmentation | ✅ | `src/mask_view.py`, `src/models/st_maskview.py` |
| ② Baselines SAITS/BRITS/CSDI/GRIN* | ✅ | `src/baselines/pypots_models.py`（GRIN* = StemGNN proxy） |
| ③ 对比实验 + 可视化 | ✅ | `run_benchmark.py`, `src/viz.py` |
| ④ 时空图 GNN + temporal attention | ✅ | `src/models/st_maskview.py` + distance adjacency |
| ⑤ 下游预警 | ✅ 骨架 | `src/downstream.py`, `run_downstream.py` |
| ⑥ HF Dataset card / notebook / Kaggle | ✅ 初版 | `dataset_card/`, `notebooks/quickstart.ipynb` |
| ⑦ 与 hydro-ml-paper 联动 | 📝 | 见下方 IP 线 |

## 与 Hydro-ML-Paper 成果线

```
水文 ML IP
├── hydro-ml-paper     径流多预见期预报（Potomac/James，信息价值叙事）
└── water-ai-do-forecast 水质时空插补 + Mask-View + 低氧/超标预警（滇池）
```

统一叙事：**观测残缺下的水信息重建与预报决策**。

## 推荐实验命令

```bash
# 冒烟（快）
python run_benchmark.py --quick --skip-pypots

# 完整对比（含 SAITS/BRITS/StemGNN；CSDI 可选）
python run_benchmark.py --include-csdi

# 下游
python run_downstream.py
```

## 投稿建议期刊

1. *Water* (MDPI) — 应用向、通过率友好  
2. *Environmental Modelling & Software* — 方法+开源  
3. *Journal of Hydroinformatics* — 水信息对口  
