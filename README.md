# Dianchi Mask-View：时空水质插补与下游预警

**核心卖点：** Mask-View multi-pattern augmentation（块状缺失友好）  
**模型：** GNN 空间消息传递 + Temporal Attention  
**基线：** Mean / LOCF / Linear / SAITS / BRITS / CSDI / GRIN*(StemGNN)  
**数据：** 滇池 22 站 × 9 指标 × 2022–2024（4h，自然缺失率 ~19.8%，块状为主）

## 快速开始

```bash
cd dianchi-maskview-imputation
pip install -r requirements.txt

# 数据已在 data/raw/；国内可：
set HF_ENDPOINT=https://hf-mirror.com

# 冒烟：只跑经典基线 + MaskView
python run_benchmark.py --quick --skip-pypots --epochs 8

# 完整对比
python run_benchmark.py --epochs 20

# 下游预警（DO骤降 / TN超标）
python run_downstream.py
```

## 代码结构

```
src/
  mask_view.py          # ① Mask-View 模式库 + 多视图损失
  models/st_maskview.py # ④ 时空图插补器
  baselines/            # ② 简单基线 + PyPOTS 封装
  data.py / metrics.py / viz.py / downstream.py
run_benchmark.py        # ③ 完整对比实验
run_downstream.py       # ⑤ 插补→预警
dataset_card/           # ⑥ HF card
notebooks/quickstart.ipynb
ROADMAP.md              # ⑦ 与 hydro-ml-paper 联动叙事
```

## Mask-View 一句话

对同一时空窗口采样 K 个互补掩码视图（point / block_time / sensor / station），用 **重建损失（人工掩蔽且原本可见的位置）+ 跨视图一致性损失** 训练，专门对抗自动站块状缺失。

## 论文级全表（已跑通）

```bash
python run_benchmark.py --epochs 40 --pypots-epochs 20   # 5 patterns × 3 rates
python export_paper_tables.py
python build_method_experiments_docx.py
```

| 总均 MAE | DO 均 MAE | Hard-pattern 均 MAE |
|---------:|----------:|--------------------:|
| **MaskView-ST 0.388** | **0.480** | **0.411** |

### 消融 + 空间加强（已完成）

```bash
python run_ablation.py --epochs 30      # 初版
python run_lambda_ablation.py           # 修复 L_cons 后重跑 λ
python finalize_ablation.py             # 汇总
# 推荐配方：
# checkpoints/maskview_spatial_plus_l0.pt  (MixHop + k=8 + station boost + λ=0)
```

| 发现 | 结论 |
|------|------|
| λ_cons | **λ=0 最优**；λ↑ 总 MAE 变差 |
| point_only | point 很强，但 sensor/station **崩** → 证明 pattern bank 必要 |
| spatial_plus_l0 | 总 MAE **0.381**、sensor **优于 BRITS**；station 缺口缩至 ~0.105 |

**作者：** 张森捷（Senjie Zhang），兰州大学，3079099853@qq.com  

**最终投稿稿（Water / MDPI）：** `paper/Water_MaskView_Dianchi_manuscript.docx`  
（若该文件被 Word 占用，用 `paper/Water_MaskView_Dianchi_manuscript_v2.docx`）  
附 `paper/cover_letter.txt`、`paper/highlights.txt`、`paper/SUBMISSION_CHECKLIST_WATER.md`

```bash
python build_water_manuscript.py
```

项目页：https://az0998.github.io/dianchi-maskview-imputation/  
仓库：https://github.com/Az0998/dianchi-maskview-imputation

英文 Method/Experiments 草稿：`paper/Method_Experiments.md` / `.docx`

## 结果输出

- `results/tables/imputation_benchmark.csv`（105 行全表）
- `results/tables/table_mae_full.csv` 等透视表
- `results/figures/fig_imputation_*.png`
- `checkpoints/maskview_st.pt`（权重不入库，`checkpoints/` 需按上述命令本地训练生成）
