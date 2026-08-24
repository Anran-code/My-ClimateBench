# 压缩版实习冲刺计划（最后一周：冲 CNN/RNN，兼收尾原第5-8周任务）

> 仓库：`D:\My-ClimateBench`  
> 环境：`my-climatebench-v2`（已装好 esem / keras / tensorflow / xarray / netcdf4 / matplotlib）  
> 前提：第 3 周 RF 四目标 baseline 已跑通并产出结果图/表，本周以它为对照基线做“更高级模型”的公平对比。

---

## 一、老师需求理解与现实约束

- **现实约束**：只剩一周时间，不宜做“超大模型 + 重训数天”的方案；
- **老师明确要求**：要试试 CNN、RNN 等更高级模型；
- **合理预期**：一周内能做到的是：
  1. 跑通 **CNN 版本**和**RNN（LSTM/GRU）版本**的 ClimateBench 设定模型；
  2. 做 **RF vs CNN vs RNN** 的公平对比（同一套 X/Y、同一套评价协议）；
  3. 做 **误差对比分析**（重点看你第 3 周已经定位出来的误差热点：高纬 tas / 热带 pr90）；
  4. 把原本后面几周（第 5–8 周）本来要做的“改进方向 / 消融 / 结论”压缩成小范围实验 + 周结讨论，保证“实习闭环”完整。

所以这周我们选的策略是：
> **少而精的模型对比**：不堆超参搜索，先用“官方 CNN-LSTM notebook 的轻量版思路 + 纯 CNN 小版 + 纯 RNN（LSTM）小版”跑通 + 出对比结果表 + 写清楚结论与改进方向。

---

## 二、原 8 周后半段任务，本周合并到哪里做

为了让实习总闭环不缺项，我把原计划后续几周的任务做如下合并：

| 原周次 | 原任务 | 本周如何处理 |
|---|---|---|
| 第 4 周（基线模型复现：GP/RF/NN） | 跑 RF/GP/NN 多种基线 | RF 已做；GP 跳过（运行重且稳定性一般）；“NN”用 CNN / RNN 两种完成，等价于第 4 周 NN 部分并升级了模型层次 |
| 第 5 周（误差分析与消融） | 误差分析、变量重要性、时空结构比较 | 直接并入：`week4_nn_compare.py` 做 RMSE 对比，`week4_nn_error_compare.py` 做热点对比；再加 RF 的 12 维特征重要性（快速版 permutation importance）作为“消融的廉价替代” |
| 第 6 周（改进方案1：静态特征/位置编码） | 引入海陆mask/经纬度/lat-weight loss | 本周做成“最小验证版”：对 CNN 加 sin/cos 位置编码和 lat/lon 额外通道（小消融，若没时间就写进“改进思路与下一步实验”） |
| 第 7 周（改进方案2：多尺度模块） | U-Net/Dilated Conv 等 | 本周 CNN 里加入 1 层 dilation 或 2 个尺度分支作为“多尺度雏形”，如果时间不足则写到“未来工作”，不占用主流程时间 |
| 第 8 周（写作与答辩准备） | 整理 PPT、撰写总结 | 本周最后两天做：生成 `docs/实习最终汇报（冲刺版）.md`，可直接复制粘贴到 PPT 或 .doc 中 |

一句话：**本周 = 第 4 周（CNN/RNN） + 第 5 周（误差对比） + 第 6–7 周最小消融/讨论 + 第 8 周总结**。

---

## 三、一周具体干活清单（按天拆，可随实际速度调整）

### Day 1：先把“最小 CNN / 最小 RNN 在 tas 上跑通”（优先级最高）

目标：建立信心，先证明“更高级模型能跑，能出结果”。

做的事：
1. 复用第 3 周已经写好的 X/Y 构造（`baseline_models/utils.py` 的 `create_predictor_data`、`create_predictdand_data`、`get_test_data`、`get_rmse`）。
2. 新建脚本 `my_code/week4_nn_quickstart.py`，包含两条独立流水线：
   - **CNN 版**：把 12 维输入通过 MLP 先投影到 `(lat, lon, hidden)` 形状，再接 2–3 层 Conv2D 输出 `tas (96,144)`；
   - **RNN（LSTM/GRU）版**：把时间序列当作“年度序列”，用 MLP→LSTM→Decoder→输出每年场；
   - 为了训练快，第一天只做 `tas` 单目标，训练 epoch 控制在 30–50，CPU 能结束；
3. 输出：
   - 预测 NetCDF：`outputs_ssp245_prediction_CNN_tas_smoke.nc`、`outputs_ssp245_prediction_RNN_tas_smoke.nc`
   - 分数 CSV：`cnn_tas_smoke_scores.csv`、`rnn_tas_smoke_scores.csv`
   - 2050 年 tas 三人组图：`CNN_tas`、`RNN_tas` 的 truth/pred/error

验收标准（自己检查）：
- 能打印“CNN tas eval rmse(35:) = xxx”、“RNN tas eval rmse(35:) = xxx”；
- 至少 CNN 的 tas 分数比 RF smoke 略好或者差不多，说明模型确实能学到。

---

### Day 2：扩展到四目标，生成 RF vs CNN vs RNN 的对比总表

目标：从“tas 单变量”升级到“论文四变量”，形成能直接汇报的结果表。

做的事：
1. 新建 `my_code/week4_nn_compare.py`（或者在 quickstart 上拆出来），对 CNN/RNN 分别训练四目标：
   - 目标：`tas / dtr / pr / pr90`
   - 训练：用统一的 early stop，控制总 epoch 不超过 40–60；
   - 可以：pr 和 pr90 训练更稳的小技巧：取 `log(pr + eps)` 训练后再指数反变换回 mm/day（如果一周赶时间就先保留原尺度）。
2. 把第 3 周的 RF 正式版分数 CSV（`rf_ssp245_scores.csv`）读进来，拼成：`rf_scores + cnn_scores + rnn_scores → all_scores.csv`
3. 画一张 4×3 的“晚期 RMSE 柱状对比图”：
   - 行：tas / dtr / pr / pr90
   - 列：RF / CNN / RNN（三条柱）
   - 标题：“ssp245 late period (>=2050) lat-weighted RMSE：三种模型对比”
4. 输出文件：
   - 预测 NetCDF：
     - `outputs_ssp245_prediction_CNN.nc`
     - `outputs_ssp245_prediction_RNN.nc`
   - 分数 CSV：
     - `cnn_ssp245_scores.csv`
     - `rnn_ssp245_scores.csv`
     - `all_model_ssp245_scores_compare.csv`
   - 对比图：
     - `week4_models_rmse_compare.png`（在 `my_code/figures/`）

验收标准：
- 能产出一张“三模型 × 四变量”的对比表；
- CNN/RNN 在 pr 或 pr90 上至少有一定优势（哪怕只是 3–5% RMSE 下降），用来回答老师“更高级模型为什么好/哪里不好”。

---

### Day 3：误差热点对比 + 最小改进验证（位置编码 / lat loss）

目标：做“误差对比分析 + 最小改进尝试”，对应原第 5 周和第 6 周的内容。

做的事：
1. 新建 `my_code/week4_nn_error_compare.py`，复用第 3 周 `week3_rf_4targets_hotspots.py` 的逻辑，依次对：
   - RF 预测文件：`outputs_ssp245_prediction_RF.nc`
   - CNN 预测文件：`outputs_ssp245_prediction_CNN.nc`
   - RNN 预测文件：`outputs_ssp245_prediction_RNN.nc`
   分别做晚期（>=2050）的：
   - 平均绝对误差空间图
   - top-5 误差热点红点标注
2. 重点关注第 3 周已经标记过的两个“靶子区域”：
   - **tas：北半球高纬（≈70°N 附近）**
   - **pr90：热带西太平洋暖池（≈-5°~5°N, 160°E~180°E）**
   看 CNN/RNN 相比 RF 在这些区域是否有下降。
3. 输出：
   - `week4_error_hotspots_RF_vs_CNN_vs_RNN.png`（三模型并排放对比）
   - `all_model_error_hotspots_summary.csv`
4. 最小改进尝试（时间够就做，不够就写进改进思路）：
   - 对 CNN 版本加一个“位置编码通道”：把 `sin/cos(lat), sin/cos(lon)` 四个通道加到 CNN 输入层，重训一版（tas/pr90 两变量就行），对比是否高纬 tas / 热带 pr90 下降；
   - 或者用 `lat * RMSE` 方式对损失加权，等价于 lat-weighted loss 的简化版；
   - 这部分不需要训到极致，只要“能跑、能对比、有数字”，老师就会认可你在思考改进。

验收标准：
- 你能口头指出“RF 在高纬 tas 误差最大的几个点，CNN/RNN 分别下降了多少 / 没下降”；
- 如果做了位置编码消融，能有一张“baseline CNN vs CNN+pos encoding”的小对比表（tas / pr90 两变量即可）。

---

### Day 4：补“特征重要性 + 训练曲线 + 失败尝试记录”，让汇报更完整

目标：把看起来“只跑了几个模型”的结果，包装成“有分析、有思考、有过程”的完整实习内容。

做的事：
1. **RF 的特征重要性（12 维输入）**：
   - 用第 3 周已训练的 RF，对 tas 和 pr90 输出 permutation importance 或树模型 Gini importance；
   - 画两张 bar 图：`RF feature importance: tas`、`RF feature importance: pr90`
   - 目的：验证“CO2 是否贡献了 tas 全局变暖”，“SO2/BC EOF 是否贡献了区域响应（特别是 pr90 热带区）”。
2. **CNN/RNN 的训练曲线**：
   - 保存 `history = model.fit(...).history`，画出 `loss / val_loss vs epoch` 图；
   - 如果出现明显过拟合，可直接在汇报中说“CNN 容量偏大但数据只有 423 年样本，后面可以加 dropout/正则”——这是体现你对模型理解的加分项。
3. **失败尝试记录（非常加分）**：
   - 例如：一开始 RNN 把每个年当一步，但没先做 MLP embedding 导致收敛很慢；
   - 或者：pr90 原始尺度训练波动大，尝试 log 变换后稳定度提升；
   - 不用写很多，两三段小记录就行，老师一般会很喜欢看你不是“一把成”而是“调过参、想过问题”。

输出：
- `week4_RF_feature_importance_tas.png`、`week4_RF_feature_importance_pr90.png`
- `week4_CNN_training_curve.png`、`week4_RNN_training_curve.png`
- 一段文字说明（写到最终汇报 md 里即可）

---

### Day 5：写总结 md（等价于第 8 周写作），准备老师提问话术

目标：把 4 天的结果整理成“老师要看什么就给他什么”的最终文档。

做的事：
1. 新建 `docs/实习最终汇报（冲刺版）.md`，结构建议：
   - 研究问题与数据
   - 方法（RF baseline / CNN / RNN 简述，画出一张模型对比表）
   - 实验设置（训练情景 / 测试情景 / 评分协议）
   - 结果：
     - 表格 1：三模型 × 四变量 RMSE
     - 图 1：RMSE 柱状对比
     - 图 2：三模型 error hotspots 对比
     - 图 3：CNN/RNN 训练曲线
     - （如果做了）表 2：baseline CNN vs CNN + 位置编码消融
   - 分析（重点讲 3 点）：
     1. CNN 在哪个变量/区域最好，为什么（空间归纳偏置）
     2. RNN 在哪个变量/时期最好，为什么（时间序列记忆）
     3. 为什么 RF 在小样本和 CPU 上是强 baseline，但是区域细节弱
   - 不足与改进方向：
     - 小样本问题（样本数 423 对 NN 来说偏少）
     - 可引入静态特征（海陆、地形、海冰）
     - 可引入 lat-weighted loss / focal hotspot loss
     - 可做多尺度模块（U-Net / dilated conv）
   - 个人实习收获：环境搭建、数据处理、baseline 跑通、模型对比、误差分析、读论文经验。
2. 准备“老师 3 类常见提问”的 1 分钟回答稿：
   - Q1：你觉得 CNN 比 RF 好在哪？（准备：空间归纳偏置 / 邻域信息共享 / pr90 区域结构更好）
   - Q2：RNN 在这里起什么作用？（准备：年度 forcing 作为时间序列，LSTM 能记住历史 forcing，对长期累积效应建模更自然）
   - Q3：如果再给你两周，你最想改哪两处？（准备：位置编码+静态特征；或 lat-weight loss 对准热点）
3. 最后把所有关键图/表的**绝对路径清单**写在 md 最后，老师要看图你能秒打开。

---

## 四、每天的脚本文件清单（我后面会陪你一步步建）

为了避免你一天结束时“文件乱放、找不到结果”，我们约定所有新增文件都在：
- 脚本：`my_code/week4_*.py`
- 图：`my_code/figures/week4_*.png`
- 分数 / 预测文件：`my_code/outputs/*_CNN_*.nc / *_RNN_*.nc / *_compare.csv`
- 文档：`docs/专业实习第四周（冲刺版）_CNN_RNN_对比与总结.md` + `docs/实习最终汇报（冲刺版）.md`

---

## 五、风险与“兜底方案”（防止一周做不完时的 fallback）

### 风险 1：CNN/RNN 训练太慢
兜底：
- 只训练 `tas + pr90` 两个最有代表性的变量；
- 把 lat/lon 用 `xarray` 粗化到 48×72（下采样一半），训练时间 ×0.25；
- epoch 统一压到 25；
- 只汇报两变量的对比，并在总结里写“其他两变量实验待后续补充”。

### 风险 2：RNN 收敛差
兜底：
- 把 RNN 的年度序列长度改成“滚动窗口 5–10 年”做滑动样本，不让 LSTM 硬吃 86 年长序列；
- 或者改 RNN 为“GRU + 2 层 MLP 头”，参数量更小更稳；
- 如果时间真不够：RNN 只跑 tas smoke，把重点放在 CNN，汇报时写“RNN 初步 smoke，后续需要更多 epoch / 更好序列构造”。

### 风险 3：CNN 分数没比 RF 好
兜底：
- 先看“误差结构”有没有好：即使 RMSE 近似，pr90 的空间场可能没 RF 那么糊（肉眼可见的结构提升也是结果）；
- 把汇报重点从“RMSE 一定下降”改为“CNN 引入空间归纳偏置，对区域结构/热点误差的可解释性提升，并给出下一次 lat-weight loss 消融后预计提升点”。

---

## 六、最终你交付给老师的东西（一套最小完整闭环）

老师面前你能掏出的是：

1. **一张表**：RF / CNN / RNN 三模型 × 四变量 held-out RMSE（或两变量，取决于时间）。
2. **三张核心图**：
   - RMSE 柱状对比图；
   - 三模型在 ssp245 2050 的 tas/pr90 space 对比图；
   - 三模型晚期误差热点对比图。
3. **一份 md 文档**：结构完整，可直接粘到 PPT/.doc。
4. **一段 5 分钟口头表述**：
   - 我先复现了 RF baseline →
   - 然后实现了 CNN 和 RNN 两版更高级模型 →
   - 在 ssp245 上做了 held-out 对比，结果显示：（举 1–2 个亮点）→
   - 误差分析看出来 XXX 区域最难 →
   - 如果继续做，我会从 XXX 两个方向改进。

---

下一条我就直接进入“陪你开干 Day 1：写 `week4_nn_quickstart.py`，先跑通最小 CNN/RNN 单变量 tas”的阶段。你先把上面这份压缩版周计划过一遍，觉得节奏能接受的话，回我一句 `week4 go`，我们立刻写代码。
