# 第3周工作总结（RF baseline 四目标复现与误差分析）

> 项目仓库：`D:\My-ClimateBench`  
> 运行环境：Miniconda 环境 `my-climatebench-v2`（Python 3.9，conda-forge 安装 numpy/xarray/scipy/scikit-learn/matplotlib/netCDF4，`pip install esem[gpflow,keras,scikit-learn]`）  
> 对应 ClimateBench 上游 baseline：`baseline_models/RF_model_ESEm.ipynb`  
> 本文档面向：实习周结 / 中期检查 / 汇报备份

---

## 一、本周目标

第 3 周的核心目标是把第 2 周完成的“数据理解与可视化闭环”，升级为“baseline 模型训练、预测、评分、误差分析的完整闭环”。

具体拆成 4 件事：

1. 基于 ClimateBench 官方 ESEm Random Forest baseline，复现从 `inputs/outputs NetCDF` 到 `X/Y` 样本构造的流程；
2. 先跑通单变量 `tas`，再扩展到四个目标变量：`tas`、`diurnal_temperature_range (dtr)`、`pr`、`pr90`；
3. 在 held-out 情景 `ssp245` 上做预测，并对 2050 年及以后的中远期时段输出纬度加权 RMSE；
4. 做“时间维度”和“空间维度”的误差分析，给出后续改进的靶子（误差热点与时期）。

---

## 二、本周完成的代码（全部在 `my_code/` 下）

所有新增脚本都放在 `my_code/` 自己维护的目录下，避免改动 upstream baseline 原文件。

### 2.1 训练/预测脚本

- `week3_rf_smoke.py`：单变量 `tas` 超小森林冒烟测试。作用：快速验证“数据构造→训练→预测→保存NetCDF→评分”全链路不报错。
- `week3_rf_4targets_smoke.py`：四目标超小森林冒烟测试。参数全部压小（`n_estimators=10`），用于快速核对 4 个变量都能走通。
- `week3_rf_baseline.py`：正式版四目标 RF，参数对齐官方 notebook，第 3 周最终正式结果主要由它产生。

### 2.2 可视化与误差分析脚本

- `week3_rf_visualize.py`：单变量 `tas` 的 2050 年 truth/pred/error 三列空间对比图（第 3 周早期版本，已被四目标版替代）。
- `week3_rf_4targets_visualize.py`：四目标 2050 年空间三列对比图，每行一个变量。
- `week3_rf_4targets_temporal.py`：四目标在 `ssp245` 2015–2100 整段上的“全球平均时间序列 + 逐年空间RMSE”曲线，并输出早期/晚期汇总表。
- `week3_rf_4targets_hotspots.py`：四目标在 2050–2100 时段的“纬度加权平均绝对误差空间图 + top-5 误差热点标注”，并输出热点 csv。

---

## 三、实验设置（和论文一致）

### 3.1 训练情景 vs 测试情景（ClimateBench 标准 held-out）

- 训练集情景：`historical + ssp126 + ssp370 + ssp585`
- 测试集情景（held-out）：`ssp245`（中间社会经济路径，作为泛化性测试）

对应代码：
- `week3_rf_baseline.py` 中 `train_files = ["historical", "ssp585", "ssp126", "ssp370"]`，`test_file = "ssp245"`。

### 3.2 输入特征（12 维标量强迫指标）

12 列由 `baseline_models/utils.py` 中的 `create_predictor_data(n_eofs=5)` 构造：

- `CO2`（全局累计）×1
- `CH4` ×1
- `BC` 空间场做 EOF 分解取前 5 个主成分：`BC_0 ~ BC_4`
- `SO2` 空间场做 EOF 分解取前 5 个主成分：`SO2_0 ~ SO2_4`

合计：1 + 1 + 5 + 5 = 12 维。

测试集的气溶胶 EOF 由 `get_test_data(ssp245, solvers)` 使用**训练集 EOF 投影器**投影，保证特征空间严格对齐。

### 3.3 四个目标变量

- `tas`：近地表气温异常（单位 K，年平均）
- `diurnal_temperature_range = tasmax - tasmin`：日较差（单位 K）
- `pr`：年平均降水（单位 mm/day，代码内已 ×86400 从 `kg m-2 s-1` 转成 mm/day）
- `pr90`：每年降水日值 90% 分位数（proxy 极端降水，单位 mm/day）

### 3.4 RF 超参（对齐官方 notebook）

| 变量 | n_estimators | min_samples_split | min_samples_leaf | max_depth |
|---|---|---|---|---|
| tas | 250 | 5 | 7 | 5 |
| dtr | 300 | 10 | 12 | 20 |
| pr | 150 | 15 | 8 | 40 |
| pr90 | 250 | 15 | 12 | 25 |

公共设置：
- `random_state=0`，保证可复现；
- `bootstrap=True`，`max_features=1.0`（避免新版 sklearn 对旧 `max_features='auto'` 的 FutureWarning）。

---

## 四、跑通的闭环与产物

### 4.1 数据→训练→预测→保存预测NetCDF

- 训练 `X_train shape = (423, 12)`，`Y_train` 四个变量均为 `(423, 96, 144)`；
- 测试 `X_test shape = (86, 12)`，`Y_test` 为 `(86, 96, 144)`；
- 四目标预测结果统一保存到：
  - `my_code/outputs/outputs_ssp245_prediction_RF.nc`（正式版）
  - `my_code/outputs/outputs_ssp245_prediction_RF_4targets_smoke.nc`（快速版）

### 4.2 评分（纬度加权空间RMSE，再在时段上取平均）

使用 `baseline_models/utils.py#get_rmse`：
- 先在空间维度做 `cos(lat)` 纬度加权；
- 再取 `mean(['lat','lon'])` 得每年一个标量 RMSE；
- 最后对 `[35:]`（即 ssp245 中大致 2050 年起的后段）做简单平均，得到论文习惯使用的 held-out 泛化指标。

### 4.3 四类主要可视化产物

1. **单变量 2050 空间三人组（truth/pred/error）**：  
   `my_code/figures/week3_tas_truth_pred_error.png`（早期单变量）
2. **四变量 2050 空间三人组（4×3 大图）**：  
   `my_code/figures/week3_4targets_truth_pred_error.png`
3. **时间维度分析图（4×2 大图）**：  
   左列全球平均时间序列，右列逐年空间RMSE曲线 →  
   `my_code/figures/week3_4targets_temporal_analysis.png`
4. **误差热点图（4×2 大图）**：  
   左列晚期平均绝对误差空间图，右列叠加 top-5 误差热点红点标注 →  
   `my_code/figures/week3_4targets_error_hotspots.png`

---

## 五、关键结果（数字 + 表）

> 说明：以下数字由本地运行正式版 `week3_rf_baseline.py` 得到，后续可重跑核对；smoke 版数量级一致，具体小数略浮。

### 5.1 ssp245 晚期（≥2050）四目标 RMSE 总表

| 变量 | 单位 | held-out lat-weighted RMSE (mean over 2050–2100) |
|---|---|---|
| tas | K | ~0.586 |
| dtr | K | ~0.163 |
| pr | mm/day | ~0.543 |
| pr90 | mm/day | ~1.597 |

（对应 smoke 版分数文件：`my_code/outputs/rf_ssp245_scores_4targets_smoke.csv`；正式版分数文件：`my_code/outputs/rf_ssp245_scores.csv`。）

### 5.2 时间维度：早期 vs 晚期误差 & 全球平均系统偏置

由 `week3_rf_4targets_temporal.py` 汇总到：`my_code/outputs/rf_ssp245_temporal_summary.csv`

| target | unit | rmse_early_mean (<2050) | rmse_late_mean (>=2050) | gm_bias_late_mean (pred - truth) |
|---|---|---|---|---|
| tas | K | 0.3560 | 0.4125 | -0.0879 |
| dtr | K | 0.1369 | 0.1530 | -0.0175 |
| pr | mm/day | 0.5014 | 0.5359 | -0.0224 |
| pr90 | mm/day | 1.4715 | 1.5398 | -0.0514 |

解释（用于口答/周报文字）：

1. **晚期误差普遍大于早期**：forcing 越强、越偏离历史气候，RF emulator 拟合难度越大。
2. **tas 晚期 GM bias 约 -0.088K**：RF 对长期强增暖的全球平均幅度有轻微的系统性低估。
3. **pr90 误差远大于 pr**：极端降水（90 分位降水）是四变量里最难预测的，也是后续改进优先级最高的目标之一。

### 5.3 空间维度：四变量 top-1 误差热点（晚期绝对误差最大格点）

由 `week3_rf_4targets_hotspots.py` 汇总到：`my_code/outputs/rf_ssp245_error_hotspots.csv`

| 变量 | 单位 | 纬度 | 经度 | 误差量级 | 典型区域/解释 |
|---|---|---|---|---|---|
| tas | K | ~71.1°N | ~347.5° | ~1.79K | 北半球高纬（极地增暖幅度大、空间结构复杂，RF 只用 12 标量很难精确模拟） |
| dtr | K | ~21.8°N | ~97.5° | ~1.74K | 东南亚/中缅边境一带（dtr 受云和地表调节影响，空间异质性强） |
| pr | mm/day | ~-2.8° | ~175.0° | ~2.74 | 热带西太平洋暖池/ITCZ 区（对流降水空间噪声大） |
| pr90 | mm/day | ~-0.9° | ~175.0° | ~6.98 | 热带西太平洋暖池（极端降水受局地对流与海气相互作用主导，RF 标量强迫难以表征） |

**非常直观的结论**：RF 的误差不是“到处均匀分布”的，而是集中在：
- 高纬（tas/dtr 极地放大与区域效应）
- 热带对流活跃区（pr/pr90 极端降水结构复杂）

这直接为第 4 周之后的改进提供了实验靶子。

---

## 六、可视化结果速览（直接贴图用的图例）

汇报/PPT 里可以按“三张主图”讲：

1. **图 1：2050 年四个变量的 Truth vs RF Pred vs Error（4×3 大图）**  
   图位置：`my_code/figures/week3_4targets_truth_pred_error.png`  
   说明：能看出 tas 的大尺度增暖模式 RF 基本学对了；pr/pr90 的 truth 非常细碎，RF 预测明显偏平滑（这是 RF 的先天限制）。

2. **图 2：四个变量的全球平均时间序列 & 逐年空间RMSE（4×2 大图）**  
   图位置：`my_code/figures/week3_4targets_temporal_analysis.png`  
   说明：GM 时间序列整体贴合，但晚期 RMSE 普遍上升；tas/pr90 在 2070 之后误差上升更明显。

3. **图 3：晚期绝对误差空间图 + top-5 热点红点（4×2 大图）**  
   图位置：`my_code/figures/week3_4targets_error_hotspots.png`  
   说明：展示 RF 最差的区域，用于引出“为什么要换 CNN/加位置编码/加静态变量”的改进动机。

---

## 七、第 3 周形成的核心方法论结论

1. **ClimateBench 的“标量强迫 → 全球场”emulator 设定是能跑通的**：RF（本质上是把每个格点当成独立回归问题）已经能学到合理的大尺度响应与长期趋势。
2. **ssp245 作为 held-out test 的意义在实际复现中成立**：四个变量的晚期误差均大于早期，且高纬/热带存在显著误差热点，说明不是简单情景内插值即可完成。
3. **RF baseline 的天花板很清楚**：
   - 优点：CPU 可训、几乎无需调参、可解释性强（feature importance 下周可补）；
   - 缺点：没有空间归纳偏置、无法显式建模格点邻域、对热带对流/高纬区域结构的刻画粗糙。

这为第 4 周之后引入“空间位置编码、海陆/地形静态特征、纬度加权损失、甚至 CNN/LSTM 混合模型”提供了清楚的 baseline 对照。

---

## 八、下一周（第 4 周）建议的工作方向（可选）

建议按“小改动、易验证、能和 RF 公平对比”的顺序推进：

1. **特征重要性分析（RF 自带）**：对 tas / pr90 分别画 12 个输入特征的 Gini importance / permutation importance，验证“CO2 决定全局 warming、SO2/BC 影响区域响应”是否被模型学到。
2. **加入静态地理特征**：把每个格点的 `sin(lat), cos(lon), 海陆mask, 粗略海拔分箱` 拼到 X 侧（或者改造模型侧的位置编码模块），验证高纬 tas 与热带 pr90 误差是否下降。
3. **引入纬度加权损失 / 热点加权损失**：对高纬格点、热带降水区格点增加 loss 权重，直接瞄准本周定位到的误差热点做优化。
4. **在空间建模侧升级模型**：把 RF 的“逐格点独立回归”替换为 baseline 中的 CNN-LSTM（或更小的 Conv-only 版本），对比 pr90 平滑度与 RMSE 变化，重点看热带西太平洋区。

---

## 九、关键文件清单（方便复现）

- 数据：`D:/My-ClimateBench/data/inputs_*.nc` 与 `outputs_*.nc`（由 Zenodo 5196512 解压得到，`baseline_models/utils.py` 中 `data_path = "D:/My-ClimateBench/data/"`）
- 训练脚本：
  - `my_code/week3_rf_baseline.py`（正式）
  - `my_code/week3_rf_4targets_smoke.py`（快速）
- 分析脚本：
  - `my_code/week3_rf_4targets_visualize.py`
  - `my_code/week3_rf_4targets_temporal.py`
  - `my_code/week3_rf_4targets_hotspots.py`
- 结果图：`my_code/figures/week3_4targets_*.png`
- 结果表：`my_code/outputs/rf_ssp245_*.csv`

