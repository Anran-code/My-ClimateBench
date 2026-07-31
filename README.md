# ClimateBench Internship Project

本项目是我的本科专业实习仓库，主题为 AI 气象/气候领域论文复现与改进。当前工作以 `ClimateBench` 为基础，先完成论文理解、代码梳理、baseline 复现与评价分析，再逐步设计可解释的小改进方案。

## Project Goal

- 精读并理解 `ClimateBench v1.0` 论文与代码框架
- 跑通原始 baseline 的训练/推理/评价闭环
- 建立实验记录、结果整理和可视化分析流程
- 在 baseline 基础上尝试小规模、可验证、可解释的改进
- 为后续扩展到更偏天气预测的任务做准备

## Upstream and Paper

本仓库基于原始开源项目开展学习与复现工作：

- Upstream repository: [duncanwp/ClimateBench](https://github.com/duncanwp/ClimateBench)
- Paper: [ClimateBench v1.0: A Benchmark for Data-Driven Climate Projections](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002954)
- Dataset: [Zenodo 10.5281/zenodo.5196512](https://doi.org/10.5281/zenodo.5196512)

感谢原作者公开论文、数据处理脚本和 baseline 实现。本仓库保留对原项目与论文的引用说明，仅在其基础上开展个人实习研究工作。

## Current Scope

当前重点围绕以下几部分展开：

- 论文分章节阅读与任务理解
- 仓库结构梳理与关键代码定位
- `Random Forest`、`Gaussian Process`、`Neural Network` 等 baseline 复现
- `SSP245` 情景下的评价指标理解与结果对比
- 后续改进点设计，例如输入增强、损失设计、评价可视化等

## Repository Structure

```text
.
├─ analysis_notebooks/   # 论文图表、综合评价、CMIP6 对比分析
├─ baseline_models/      # 各类 baseline notebook 与公共工具
├─ docs/                 # 实习过程文档、周报与阶段性记录
├─ prepare_data.py       # 原始 NorESM2 输出整理脚本
├─ prep_input_data.ipynb # benchmark 输入输出构造脚本
├─ inputs_NorESM2_ERF.csv
└─ README.md
```

## Baseline Overview

原始仓库包含多类 baseline，后续复现工作将围绕这些模型逐步推进：

- Pattern Scaling
- Random Forest
- Gaussian Process
- Neural Network

论文与原仓库中常见的核心预测变量包括：

- `tas`
- `diurnal_temperature_range`
- `pr`
- `pr90`

## Environment Notes

原始项目依赖 `ESEm` 以及若干科学计算库。原 README 给出的基础安装说明如下：

```bash
conda install -c conda-forge iris
pip install esem[gpflow,keras,scikit-learn] eofs
```

后续我会根据自己的复现实验补充更完整的环境配置说明、运行顺序和注意事项。

## Progress

当前已完成或正在进行的工作：

- [x] 建立个人实习仓库
- [x] 切换项目工作区到个人仓库
- [x] 开始论文分章节精读
- [x] 初步梳理仓库结构与 benchmark 逻辑
- [ ] 规范化 README 与项目文档
- [ ] 跑通至少一个 baseline 闭环
- [ ] 补充实验记录与结果分析
- [ ] 设计并验证一个小改进点

## Notes

- 本仓库主要用于课程/实习研究与学习记录
- 大体积数据文件原则上不直接上传到 GitHub
- 若复现实验与原仓库实现存在差异，会在后续文档中单独说明

