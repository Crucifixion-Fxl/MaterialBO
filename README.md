# MetalBayes

三目标贝叶斯优化框架，用于在**粘附 (Adhesion)**、**覆盖 (Coverage)**、**均匀性 (Uniformity)** 三个目标下优化配方参数。支持有机物 (organic) 与氧化物 (oxide) 两类配方的独立优化。**后续将优化为专用于材料领域的贝叶斯优化库。**

## 功能特点

- **三目标贝叶斯优化**：基于 BoTorch，使用高斯过程代理与 EHVI 类采集函数
- **双配方支持**：有机物（有机配方、浓度、温度、浸泡时间、pH、固化时间等）与氧化物（金属类型、浓度、摩尔比等）两套参数空间与约束
- **多种目标函数**：`simple` / `complex` / `standard`（Botorch DTLZ2）/ `paper`（论文多项式），可加高斯噪声
- **约束与离散化**：pH 安全约束、氧化物约束等，支持离散/步长参数
- **超体积与 Pareto 前沿**：记录每轮超体积、导出 Pareto 解与优化历史

## 环境要求

- Python 3.8+
- PyTorch
- BoTorch
- GPyTorch
- NumPy, Pandas

```bash
pip install torch botorch gpytorch numpy pandas
```

## 项目结构

```
MetalBayes/
├── src/                    # 核心库
│   ├── optimizer.py        # 优化器 (OrganicOptimizer, OxideOptimizer)
│   ├── surrogate.py        # 高斯过程代理
│   ├── acquisition.py      # 采集函数 (EVHI)
│   ├── objective_functions.py  # 目标函数
│   └── constraints.py      # 约束处理
├── example_usage.py        # 交互式示例：选择有机/氧化物、目标版本、迭代数、噪声
├── examples/
│   └── multi_objective_bo.py   # BoTorch 多目标示例
├── analysis/
│   ├── compare_predictions.py  # 代理模型预测 vs 真实目标值对比
│   └── README.md
├── output/                 # 优化结果（按 organic/oxide、版本、时间戳组织）
├── runs/                   # 运行脚本目录
└── test/
    └── test_regressors.py
```

## 快速开始

### 交互式运行

```bash
python example_usage.py
```

按提示选择：

1. **优化类型**：有机物 / 氧化物 / 全部  
2. **目标函数版本**：simple / complex / standard / paper  
3. **迭代次数**（默认 200）  
4. **噪声水平**（默认 0，可设 >0 做带噪声优化）

结果会写入 `output/organic/` 或 `output/oxide/` 下对应子目录。

### 代码调用示例

```python
from src.optimizer import OrganicOptimizer, OxideOptimizer

# 有机物优化（需自行构造 param_space，参见 example_usage.py）
optimizer = OrganicOptimizer(
    param_space=param_space,
    output_dir="./output/organic/paper/run_001",
    seed=42,
    objective_version="paper",   # simple / complex / standard / paper
    noise_level=0.0
)
optimizer.optimize(n_iter=200, simulation_flag=True)
pareto_x, pareto_y = optimizer.get_pareto_front()
```

氧化物优化同理，使用 `OxideOptimizer` 与对应的 `param_space`。

## 目标与输出

- **三目标**：Adhesion、Coverage、Uniformity（均最大化，内部会做适当变换）
- **输出文件**（在 `output/` 下每次运行的目录中）：
  - `optimization_history.json`：每轮超体积、采集值等
  - `experiment.csv`：参数与三目标值、时间戳等

## 分析脚本

`analysis/compare_predictions.py` 用于对比**代理模型预测**与**真实目标值**（从 CSV 读取），并生成 R²、MAE、残差与曲线对比图。详见 `analysis/README.md`。

## 测试

```bash
pytest test/
```

## 后续规划

本项目将逐步优化为**专用于材料领域的贝叶斯优化库**，面向材料配方、工艺参数等多目标优化场景，提供更贴合材料实验与模拟的接口与功能。

## 开发进度表

| 阶段 | 任务 | 优先级 | 状态 |
|------|------|--------|------|
| **基础建设** | 固定依赖版本，添加 `requirements.txt` 或 `pyproject.toml` | 高 | 待办 |
| | 统一日志与异常处理，便于调试与集成 | 中 | 待办 |
| | 完善单元测试，覆盖优化器、代理、约束等核心模块 | 高 | 待办 |
| **库化与发布** | 规范包结构，支持 `pip install -e .` 安装 | 高 | 待办 |
| | 定义稳定公共 API，区分内部实现与对外接口 | 高 | 待办 |
| | 发布到 PyPI 或私有索引，便于他人安装使用 | 中 | 待办 |
| **材料领域专用** | 抽象「材料配方」与「工艺参数」通用数据模型，便于扩展新配方类型 | 高 | 待办 |
| | 支持从配置文件（YAML/JSON）加载参数空间与约束，减少硬编码 | 中 | 待办 |
| | 预留与实验/模拟软件对接接口（如回调、结果文件解析） | 中 | 待办 |
| | 增加更多材料相关目标与约束（如成本、稳定性、环保指标等） | 中 | 待办 |
| **文档与示例** | 编写 API 文档（如 Sphinx + autodoc） | 中 | 待办 |
| | 提供「从零运行」教程与典型材料优化案例 | 中 | 待办 |
| | 在 README 中补充引用与论文链接（若有） | 低 | 待办 |
| **长期** | 支持异步/批量实验（多次采样后统一评估） | 低 | 待办 |
| | 支持多保真、迁移学习等进阶贝叶斯优化能力 | 低 | 待办 |

完成一项后可将对应行的「状态」改为「进行中」或「完成」。

## 许可证

请根据项目实际情况添加许可证信息。
