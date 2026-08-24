# 专业实习最终汇报（冲刺版）：ClimateBench baseline RF vs CNN vs RNN 对比与改进方向

> 适用：最后一周实习结项 / 老师提问准备 / 后续 PPT 内容骨架  
> 仓库：`D:\My-ClimateBench`  
> 前置工作：第 1–3 周已完成环境搭建、数据理解与可视化、RF baseline 四目标复现；第 4 周冲刺 CNN/RNN 并做三模型对比。

---

## 0. 一句话研究问题

在 ClimateBench v1.0 设定下，给定若干强迫情景下的 12 维年度强迫指标（CO2/CH4 + BC/SO2 的 EOF 主成分），建立一个便宜的 emulator，去预测 ssp245 等未见情景下的四个年度气候变量场（`tas / dtr / pr / pr90`），并比较三类模型：**Random Forest (RF) / CNN / RNN** 的数值表现与空间结构误差。

---

## 1. 数据与实验设置

### 1.1 数据来源（第 2/3 周已完成的铺垫）

- 原始数据来源：ClimateBench 官方 Zenodo `10.5281/zenodo.5196512`，解压到 `D:/My-ClimateBench/data/`
- `prepare_data.py` 从 NorESM2-LM 拉取日值变量并年度聚合、派生 `dtr` 和 `pr90`（原始数据准备，我未重跑，仅理解链路）
- `prep_input_data.ipynb` 把 forcing 和输出整理成 `inputs_*.nc` 和 `outputs_*.nc`（我使用其产物）
- `baseline_models/utils.py` 提供四个核心函数：`create_predictor_data / get_test_data / create_predictdand_data / get_rmse`

### 1.2 训练-测试拆分（与 ClimateBench 论文一致）

- 训练情景：`historical + ssp126 + ssp370 + ssp585`
- 测试情景（held-out）：`ssp245`
- 评价时段（论文习惯）：取 `ssp245` 的后段（大致 2050–2100，对应时间索引 `[35:]`）做 RMSE 汇总

### 1.3 输入输出定义

- 输入：`X shape = (N_sample, 12)`，12 维标量强迫
  - `CO2, CH4`
  - `BC_0..BC_4`（BC 空间场做 EOF 前 5 个主成分）
  - `SO2_0..SO2_4`（SO2 空间场做 EOF 前 5 个主成分）
  - 注意：测试集气溶胶必须用训练集的 `eof_solvers` 做投影，保证特征一致
- 输出：四变量全球 96×144 网格年度异常场
  - `tas`: 2m 气温异常 (K)
  - `dtr = tasmax - tasmin`: 日较差 (K)
  - `pr`: 年平均降水 (mm/day)
  - `pr90`: 年降水 90 分位 (mm/day)

### 1.4 评价指标

ClimateBench 风格的“纬度加权空间 RMSE”：

1. 每个时间步先算 `(pred - truth)^2` 在空间上的 `cos(lat)` 加权平均
2. 再开方得到逐年空间 RMSE（`baseline_models/utils.py#get_rmse`）
3. 最后对 2050–2100 时段再做简单平均，得到一个“晚期 held-out RMSE 标量”用于横向对比

我们所有模型和所有变量都严格用同一套评分协议，确保公平对比。

---

## 2. 三种模型的实现与关键思想

### 2.1 RF baseline（第 3 周已跑通）

- 代码入口：`my_code/week3_rf_baseline.py`
- 实现：使用 ESEm 的 `rf_model`，本质上是把 96×144 每个格点当成一个独立回归目标，用 12 维 X 去拟合。
- 主要优点：
  - CPU 可训、速度快、稳定性强；
  - 几乎无需调参就能给出合理的大尺度均值响应；
  - 可解释性好（树模型的 feature importance，原第 5 周任务）。
- 主要缺点：
  - 无空间归纳偏置，格点之间无信息共享；
  - 对高纬区域结构和热带降水极值容易“学得太糊”。

### 2.2 CNN emulator（本周 Day2 核心任务）

- 代码入口：`my_code/week4_nn_compare.py` 内 `build_cnn()`
- 结构流程（一句话版）：
  1. 12 维标量输入通过 MLP 映射到 `(24, 36, 16)` 的低分辨率 feature map
  2. 两层 Conv2DTranspose(×2 upsampling) 把空间分辨率放大到 `(96, 144)`
  3. 中间穿插普通 Conv2D 做空间邻域特征融合
  4. 最后 1×1 Conv 输出单通道 `tas / dtr / pr / pr90` 场
- 关键设计：
  - Loss：用自定义 `lat_weighted_mse`（和评分指标保持一致），这是原第 6 周“纬度加权损失”的最小实现；
  - 上采样 + Conv 的组合给模型注入空间归纳偏置：邻域像素共享卷积核，学习平滑的区域结构。
- 期望强项：
  - `pr90` 和高纬 `tas` 的空间结构比 RF 平滑合理；
  - 与第 3 周 RF 误差热点对比能看下降（我们 Day3 专门做图验证）。

### 2.3 RNN（LSTM）emulator（本周 Day2 + RNN 部分）

- 代码入口：`my_code/week4_nn_compare.py` 内 `build_rnn()`
- 序列构造：
  - 取 `seq_len=11` 年滑动窗口，即每个样本输入过去 11 年的 forcing 序列，预测窗口最后一年对应的全球场；
  - 训练集/测试集分别用滑动窗口构造成 `(N_sample, 11, 12)` 形状的序列输入。
- 结构：
  - 11×12 序列 → Dense embedding → 两层 LSTM（第二层 return_sequences=False）
  - LSTM 输出 → MLP → 低分辨率 feature map → 同样的 2× upconv + conv 解码到 96×144。
- 关键思想：
  - 气候系统有记忆，过去若干年的 forcing 累积会影响当年响应；
  - RNN 的作用是在“强迫时间维”做 temporal inductive bias，和 CNN 的空间归纳偏置互补。
- 期望强项：
  - 晚期（2070+）趋势性强的 `tas` 或 `dtr` 系统偏置更小；
  - 如果观察到 CNN 晚期系统偏负，RNN 可能缓解（原第 5–6 周“系统偏置分析”的结论点）。

---

## 3. 主要结果表与图（实验跑完成后直接填数值）

### 3.1 ssp245 晚期 held-out RMSE 对比表（核心交付表 1）

> 说明：下为模板占位格式，实际跑 `week4_nn_compare.py` 后从 `all_model_ssp245_scores_compare.csv` 抄数值即可。

| target | unit | RF RMSE | CNN RMSE | RNN RMSE | CNN imp vs RF | RNN imp vs RF | 简单结论 |
|---|---|---|---|---|---|---|---|
| tas | K | (填) | (填) | (填) | (填) | (填) | 例：RNN 在晚期略好于 CNN/RF |
| dtr | K | (填) | (填) | (填) | (填) | (填) |  |
| pr | mm/day | (填) | (填) | (填) | (填) | (填) |  |
| pr90 | mm/day | (填) | (填) | (填) | (填) | (填) | 例：CNN 空间结构更好，RMSE 相对 RF 略降 |

- 配套柱状图：`my_code/figures/week4_models_rmse_compare.png`
- 配套分数文件：`my_code/outputs/all_model_ssp245_scores_compare.csv`

### 3.2 空间误差热点跨模型对比表（核心交付表 2）

> 跑 `week4_nn_error_compare.py` 后写；关注 `tas` 高纬 #1 和 `pr90` 热带 #1 两个“重点靶子”。

| 关注点 | RF 最大误差点 (lat,lon) / 数值 | CNN 对应点数值 | RNN 对应点数值 | 观察到的变化 |
|---|---|---|---|---|
| tas 高纬 hotspot #1 | (填) | (填) | (填) |  |
| pr90 热带 hotspot #1 | (填) | (填) | (填) |  |

- 配套图：`my_code/figures/week4_error_hotspots_RF_vs_CNN_vs_RNN.png`
- 配套 csv：`my_code/outputs/all_model_error_hotspots_summary.csv`

### 3.3 训练曲线与失败/成功尝试记录（加分项，原第 7–8 周内容压缩）

- 训练曲线图：`my_code/figures/week4_cnn_rnn_training_curves.png`（tas / pr90 两个代表性变量）
- 建议口头准备的“过程记录”：
  1. CNN 第一次 reshape 时 96×144 下采样比例不对 → 改成 `n_lat//4, n_lon//4` 的精确通道数乘积；
  2. RNN 第一年用 `SEQ_LEN-1` 真值做占位，避免序列数据空缺（这是“先跑通再优化”的工程选择）；
  3. pr90 原尺度训练 loss 较大，可考虑 log(y+eps) 反变换（写进“下一步改进方向”即可）。

---

## 4. 你可以给老师讲的 3 个“结论点”

### 结论 1：更高级模型“不一定整体 RMSE 大降”，但通常在结构和重点区域有明显提升

你可以先讲数字，再讲空间图：
- 先展示 `week4_models_rmse_compare.png` 的整体 RMSE；
- 再展示 `week4_error_hotspots_RF_vs_CNN_vs_RNN.png` 的高纬 tas / 热带 pr90 热点；
- 最后一句话总结：
  “如果只看全局平均 RMSE，三者差距可能没那么夸张，但在第 3 周已经定位出来的误差热点区域，CNN/RNN 因为有空间/时间归纳偏置，会比 RF 有更清晰的改进，这就是为什么我们要做更高级模型。”

### 结论 2：CNN 擅长空间结构，RNN 擅长长期累积记忆

- CNN 对 `pr90` 和高纬 `tas` 的误差区域更集中、响应空间更平滑：体现“空间归纳偏置”价值（原第 4 周对 CNN/RNN inductive bias 的理解）。
- RNN 把 11 年 forcing 序列作为输入，显式建模过去强迫累积，如果 `tas` 晚期系统偏置下降，就是“时间归纳偏置”的直接证据。

### 结论 3：这个任务数据集样本少（423 训练年），是后面改进的主要矛盾

- RF 这种“低方差、稳定”的模型在小样本上非常强，这也是为什么 ClimateBench 把它当基线；
- CNN/RNN 一旦参数太多容易过拟合训练分布，所以可以引出“下一步要做正则化 / 静态特征增强 / 小样本策略”（自然地把原 5–7 周内容作为未来工作）。

---

## 5. 不足与改进方向（对应原第 5–7 周压缩进来的“改进实验”）

下面列 4 条最“容易解释、能讲出道理、真的可能涨点”的方向，你汇报时至少挑 2 条详细讲：

### 5.1 加入静态地理特征 / 位置编码

- 做法：对 CNN 的输入端拼 4 个静态通道 `sin(lat) / cos(lon) / land_mask / elevation_bin`（ClimateBench 有 lat/lon，海陆可从 NorESM2 网格元数据或静态场取）。
- 动机：让模型不再“对所有格点位置一视同仁”，降低高纬和热带错误（对应第 3 周热点）。
- 公平对比：保持其他结构不变，只做 +pos 消融。

### 5.2 强化 lat-weighted loss / 热点 weighted loss

- 已在 week4 CNN/RNN 用了 lat-weighted MSE；
- 下一步可加“hotspot loss”：对第 3 周输出的 top-5% 误差格点额外加权重，直接对着高纬 tas 和热带 pr90 区域打靶子（对应原第 6 周损失函数改进）。

### 5.3 对降水/极端降水做 log-变换或多尺度模块

- 降水正偏、长尾分布严重，训练中 log(pr+eps) → 预测后 `exp(y) - eps` 的回归常常更稳；
- 也可以把 CNN 改造成“多尺度结构”（两条不同 kernel size 的分支并联，或加 1–2 层 dilated conv），捕捉热带降水的不同空间尺度（对应原第 7 周多尺度模块）。

### 5.4 数据增强或正则化解决小样本问题

- 可对情景做“时间滑窗重采样”（类似 RNN 的输入构造，但让 CNN 也接受滑窗做堆叠输入）；
- 或让 CNN/RNN 增加 dropout / batchnorm / weight decay，抑制 423 样本下的过拟合。

---

## 6. 个人实习收获（最终结项 / PPT 最后一页常要写）

按周次串起来讲，简洁、真实：

1. 第 1 周：完成 conda 环境搭建，理解 ClimateBench 论文 + 数据链路（prepare_data.py → prep_input_data.ipynb → baseline_models/utils.py）；
2. 第 2 周：跑通数据读取、样本构造、EOF 压缩、基础可视化（forcing 时间序列、tas/pr 空间图）；
3. 第 3 周：复现官方 ESEm RF baseline，扩展到四目标、输出 RMSE、做时间维度（早期/晚期 RMSE + bias）和空间热点（top-5 误差区域）两套分析；
4. 第 4 周：理解 RF/CNN/RNN 三类模型的 inductive bias 差异，落地自己的 CNN emulator 和 LSTM emulator，统一用 lat-weighted MSE 做训练，并形成三模型在 ssp245 上的公平对比，最后梳理改进方向（静态特征、加权损失、多尺度结构、小样本正则）。

---

## 7. 结果图与脚本索引（汇报时秒开）

### 关键脚本（你自己跑的顺序）

1. 单变量 tas smoke（Day1 快速建立信心）：  
   `my_code/week4_nn_quickstart.py`
2. 三模型 × 四变量正式对比（Day2 主实验）：  
   `my_code/week4_nn_compare.py`
3. 三模型误差热点对比（Day3 分析）：  
   `my_code/week4_nn_error_compare.py`
4. 周计划（本周做什么/后面几周如何并入）：  
   `docs/专业实习第四周（冲刺版）_CNN_RNN对比与收尾计划.md`

### 主要产物（汇报时建议按“一张图+一句话结论”讲）

- 分数表：  
  `my_code/outputs/all_model_ssp245_scores_compare.csv`
- RMSE 柱状对比图：  
  `my_code/figures/week4_models_rmse_compare.png`
- 训练曲线：  
  `my_code/figures/week4_cnn_rnn_training_curves.png`
- 三模型误差热点总图：  
  `my_code/figures/week4_error_hotspots_RF_vs_CNN_vs_RNN.png`
- 热点 csv：  
  `my_code/outputs/all_model_error_hotspots_summary.csv`

---

## 附：老师常见 5 个提问的“1 分钟标准答案”

1. **Q：为什么 ssp245 当测试集？**  
   A：ClimateBench 要测模型对新情景的泛化能力，不是情景内插值。ssp245 介于低强迫 ssp126 和高强迫 ssp585 之间，是合适的 held-out 中间情景。

2. **Q：为什么输入是 12 维标量？**  
   A：RF baseline 官方设定是“只用强迫的全局量 + 气溶胶主模态”，这样计算便宜；后续如果要加空间特征，可以把原始 SO2/BC 2D 场或静态海陆地形场拼进去，CNN 天然能吸收这些空间输入。

3. **Q：CNN 和 RNN 的归纳偏置本质差别是什么？**  
   A：CNN 是空间归纳偏置：假设相邻格点共享相似结构，通过卷积核共享权重建模邻域关系；RNN 是时间归纳偏置：假设当前气候场依赖过去若干年强迫累积，通过 LSTM 的门控机制建模长短期记忆。

4. **Q：为什么用纬度加权 RMSE？**  
   A：同样 2.5° 网格，高纬格点实际面积远小于低纬，直接平均会夸大极地误差的全球权重；纬度加权让评分与“面积平均误差”一致，更符合地球系统习惯。

5. **Q：如果再给你两周你最想做哪两件事？**  
   A：(1) 给 CNN 加位置编码和海陆/地形静态特征（瞄准高纬 tas 和热带 pr90 热点）；(2) 引入热点加权 loss + 降水分变量 log 变换，让长尾分布和重点区域学得更准。

---

最后提醒：本 md 是“骨架模板”。等你跑完 `week4_nn_compare.py` 和 `week4_nn_error_compare.py` 后，只需把第 3 节里的表格占位用实际数值替换掉，你就有一份可直接复制到 PPT 或 .doc 的结项汇报内容。
