# 水文 ML 成果线（双项目联动）

## 定位

| 项目 | 问题 | 叙事关键词 |
|------|------|------------|
| `hydro-ml-paper` | 径流多预见期预报（Potomac/James） | 信息价值、persistence/路由基准、洪水 CSI |
| `water-ai-do-forecast` | 水质时空插补 + 预警（滇池） | Mask-View、块状缺失、邻站图、低氧/超标 |

统一 IP 句：

> 在观测不完整与简单基准对照下，量化机器学习对水文/水质**决策信息**的边际贡献。

## 可合并的对外材料

1. 个人主页 / 开题：一条时间线讲两个 case  
2. 综述短文：Hydrological forecasting vs water-quality imputation under missingness  
3. 代码 monorepo（可选）：`hydro-ml-suite/{forecast,imputation}`

## 不建议立刻做的

- 强行把两套代码揉进同一个训练脚本  
- 用同一个模型同时刷径流和水质（审稿易显拼凑）
