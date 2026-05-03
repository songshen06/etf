# `quantlab` CLI 说明

面向 **review / 对接 Agent** 的速查表。业务实现均在 `core/`（`runner` + `schemas`），CLI 仅解析参数、调用 `run_*`、打印或落盘。

## 安装与入口

```bash
cd <repo_root>
pip install -e .
quantlab --help
quantlab <subcommand> --help
```

未安装包时：

```bash
python -m cli.main --help
```

数据库默认解析顺序：环境变量 `ETF_DB_PATH`（若设置）→ 仓库根目录 `etf_data.db` → `db/etf_data.db`（与 `core.paths.resolve_db_path` 一致）。

---

## 忘记命令怎么找（推荐流程）

1) 先列出所有子命令：

```bash
quantlab --help
```

2) 再对目标子命令看参数（最常用）：

```bash
quantlab <subcommand> --help
```

3) 若你只记得“想做的事”，先看本文档的「任务速查」与「扩展子命令（策略/组合/评分）」两节；它们基本覆盖了 `quantlab --help` 的全部入口。

---

## 升级记录（CLI）

### 2026-03-28：全样本分位桶（单一数据源）

- 研究帧在 **`prepare_research_frame`** 末尾一次性写入 **`bias_bucket` / `momentum_bucket` / `volume_ratio_bucket` / `daily_change_bucket`**（全样本等频 **Q1–Q5**，与 **`--bias-q`** 过滤口径一致）。
- **`analyze-path-quality`**：`feature_breakdowns` **只按上述列分组**，不在过滤后的子样本上重新 `qcut`。例如 **`--bias-q Q3-Q4`** 时，`bias_rate` 分解里只会出现 **Q3、Q4**（及缺失时的 **NA**），不会再出现子样本重标后的 **Q1/Q2**。
- **`analyze-path-rules`**：规则里的 **Q1–Q5 / Qa-Qb** 与全样本桶一致；**`--bucket-n` 已忽略**（参数保留仅为兼容旧脚本）。
- **`backtest --bias-q`**：是否允许入场与预计算 **`bias_bucket`** 对齐，不再对乖离列临时重算五分位。
- **JSON**：`PathQualityResponse` 已移除仅作说明用的 `bias_quantile_range` 字段；语义由数据管道保证。

---

## 全局约定

| 约定                    | 说明                                                                               |
| ----------------------- | ---------------------------------------------------------------------------------- |
| `--code` / `--etf-code` | 必填，ETF 代码（如 `515080`）                                                      |
| `--db`                  | 可选，SQLite 路径；省略则用默认解析                                                |
| `--json`                | 将 **完整 Pydantic 响应** 以 JSON 打印到 stdout（便于 Agent 解析）                 |
| `--save-json PATH`      | 另存同一份 JSON 到文件（多数子命令支持，见各节）                                   |
| `--bias-q`              | 可选，如 `Q3-Q4`：按 **全样本预计算** 的 `bias_bucket` 过滤（与 path-quality / backtest 等一致） |
| 退出码                  | 成功 `0`；异常 `1`（错误信息在 stderr）                                            |

---

## 子命令一览

| 子命令                 | 作用                                                                 | 核心 API |
| ---------------------- | -------------------------------------------------------------------- | -------- |
| `health`               | 单标的：原始行校验摘要、清洗前后行数、问题列表、无效行抽样           | `run_health` → `HealthResponse` |
| `signal-research`      | 分层信号（NEG / NEG+LOW / NEG+LOW+HIGH）事件研究 + Plotly 图 JSON    | `run_signal_research` → `SignalResearchResponse` |
| `recommend`            | 规则化 ETF 适配 + 默认信号 / mode / 乖离 MA（不跑事件研究）          | `run_recommendation` → `RecommendationResponse` |
| `state-rank`           | 三维三分位全组合状态（27 档）按 horizon 胜率排名 TopK / BottomK      | `run_state_ranking` → `StateRankingResponse` |
| `analyze-transition` | 指定起点态 → 多 horizon 未来态分布（研究帧，无交易）                 | `run_state_transition` → `StateTransitionResponse` |
| `analyze-path-quality` | 起点态 → horizon 内是否达目标态；按 **全局 Q 桶** 分解特征（无交易） | `run_path_quality` → `PathQualityResponse` |
| `analyze-path-rules`   | 在起点样本上挖掘 **Q1–Q5** 连续区间规则（与全局桶一致，无交易）    | `run_path_rule_mining` → `PathRuleMiningResponse` |
| `backtest`             | 分层信号、不重叠持仓、持有期、次日开盘；可选 `--bias-q` 乖离桶过滤   | `run_backtest` → `BacktestResponse` |
| `report`               | 依次跑 health + signal-research + backtest，写 JSON/CSV/HTML/Markdown | `run_report` → `ReportResponse` |

**注意**：`signal-research` / `backtest` / `report` 使用的是 **同一套分层信号参数**（动量窗、乖离 MA、量比窗、分位、`rolling`/`full_sample`）。`state-rank` 使用 **另一套三分位划桶**（`ternary-q1`/`ternary-q2`），语义对齐原 `signal-quality-analyzer`，与分层 NEG/LOW/HIGH **不同**。

### 扩展子命令（策略 / 组合 / 评分）

这些命令也会出现在 `quantlab --help` 中，但它们不走 `core.runner` 的那套研究帧管线，而是更偏“实用工具 / 策略系统”的入口（实现主要在 `quantlab/cli/`）。

| 子命令 | 作用 | 常用场景 |
| --- | --- | --- |
| `score-bloody-chip` | 单标的「血筹」评分（可 `--json`） | 快速判断是否进入“血筹机会”区 | 
| `rank-bloody-chip` | 全库 ETF 按血筹评分排名（可去重） | 一次找出不同资产组最值得看的机会 | 
| `explain-bloody-chip` | 生成血筹解释物料到目录 | 输出图/文本用于复盘或对外展示 | 
| `dividend-signal` | 红利 ETF “低位累积”信号 + 分层加仓 + 轻减仓；可落盘状态 | 偏实盘：红利类低频操作建议 | 
| `dividend-status` | 查看 `dividend-signal` 的持仓状态文件 | 看当前 layer / avg_cost / 浮盈亏 | 
| `market-state` | 基于 `momentum_q` 识别 TREND/DOWN/RANGE | 给策略系统提供 regime 上下文 | 
| `recommend-strategy` | 按品类路由策略，给出动作/仓位/退出计划（可 `--json`） | “我现在该不该加/减/持有？” | 
| `backtest-strategy-report` | 对策略系统做回测汇总与对比（可画图到目录） | 横向比较各类策略效果 | 
| `debug-trades` | 输出策略动作时间线（BUY/ADD/REDUCE/EXIT） | 解释某段时间为什么会交易 | 
| `detect-market-regime` | 识别市场状态（AGGRESSIVE/BALANCED/DEFENSIVE） | 驱动三 ETF 组合推荐 | 
| `recommend-portfolio` | 根据市场状态输出三 ETF 目标权重 | 给出目标配置 | 
| `backtest-portfolio-regime` | regime-based 三 ETF 组合历史回测 | 检验组合框架 | 
| `recommend-portfolio-action` | 输入当前持仓，输出“本次调仓建议”（带限幅/滞后） | 实盘执行建议 | 

### 任务速查（按你想做什么）

- 数据质量体检：`quantlab health --code <ETF>`
- 看分层信号在未来 20/60/120 天表现：`quantlab signal-research --code <ETF> --horizons 20,60,120`
- 做单标的分层回测：`quantlab backtest --code <ETF> --position-rule layered`
- 一键产出研究物料包：`quantlab report --code <ETF> -o <DIR>`
- 研究“起点态 → 未来是否达目标态”：`quantlab analyze-path-quality ...`
- 挖规则（Q 区间）：`quantlab analyze-path-rules ...`
- 血筹评分（单只 / 排名 / 出解释物料）：
  - `quantlab score-bloody-chip --code <ETF>`
  - `quantlab rank-bloody-chip --top-k 10`
  - `quantlab explain-bloody-chip --code <ETF> -o artifacts/bloody_chip`
- 红利低位累积信号（会写状态到 `~/.quantlab/dividend_state.json`，可 `--dry-run`）：
  - `quantlab dividend-signal --code 515080 --db ./etf_data.db`
  - `quantlab dividend-status --db ./etf_data.db`
- 策略系统“现在怎么做”：`quantlab recommend-strategy --etf <ETF> --db ./etf_data.db --current-position 0.7`
- 三 ETF 组合（市场状态 → 目标权重 → 历史回测 → 实盘调仓建议）：
  - `quantlab detect-market-regime --db ./etf_data.db`
  - `quantlab recommend-portfolio --db ./etf_data.db`
  - `quantlab backtest-portfolio-regime --db ./etf_data.db --start-date 2020-02-06 --rebalance monthly`
  - `quantlab recommend-portfolio-action --db ./etf_data.db --current-a500 0.4 --current-dividend 0.3 --current-dividend-growth 0.3`

---

## 1. `health`

**用途**：数据质量速览（与 Streamlit「Data Health」同源）。

**参数**

| 参数              | 默认  | 说明                           |
| ----------------- | ----- | ------------------------------ |
| `--code`          | 必填  | ETF 代码                       |
| `--db`            | 自动  | 数据库路径                     |
| `--invalid-limit` | `500` | 返回的「清洗前异常行」抽样上限 |
| `--json`          | off   | 打印 `HealthResponse` JSON     |
| `--save-json`     | 无    | 写入 JSON 文件                 |

**示例**

```bash
quantlab health --code 515080 --json
quantlab health --code 515080 --db ./etf_data.db --save-json ./out/health.json
```

---

## 2. `signal-research`

**用途**：事件研究（信号日收盘 → horizon 日收盘收益）；输出各 tier 表格 + `charts[].plotly_json`。

**参数（在 `_add_signal_params` 上）**

| 参数                     | 默认        | 说明                                        |
| ------------------------ | ----------- | ------------------------------------------- |
| `--mode`                 | `rolling`   | `full_sample` / `rolling`（分位数定义方式） |
| `--bias-ma`              | `120`       | 乖离均线：须为 `60` / `120` / `250`         |
| `--momentum-window`      | `10`        | `5` / `10` / `20` / `60`                    |
| `--volume-ma-window`     | `20`        | `5` / `10` / `20` / `60`                    |
| `--quantile-low`         | `0.33`      | 分层信号 NEG/LOW 侧分位（离散可选值）       |
| `--quantile-high`        | `0.67`      | 分层信号 HIGH 侧分位                        |
| `--rolling-window`       | `252`       | `rolling` 模式下滚动长度                    |
| `--horizons`             | `20,60,120` | 逗号分隔，事件研究持有天数                  |
| `--json` / `--save-json` |             | 同全局约定                                  |

**示例**

```bash
quantlab signal-research --code 515080 --mode full_sample --bias-ma 120 --horizons 20,60 --json
```

---

## 3. `state-rank`

**用途**：动量三档（NEG/NEU/POS）× 乖离三档（LOW/MID/HIGH）× 量比三档（LOW/NORMAL/HIGH），按 **单一 `--horizon`** 前向收益统计，输出胜率 **Top / Bottom**。

**参数**

| 参数                            | 默认            | 说明                                |
| ------------------------------- | --------------- | ----------------------------------- |
| `--code`                        | 必填            |                                     |
| `--db`                          | 自动            |                                     |
| `--mode`                        | `rolling`       | 划桶用 `full_sample` / `rolling`    |
| `--bias-ma`                     | `120`           | Pydantic 校验：`60` / `120` / `250` |
| `--momentum-window`             | `10`            | `5` / `10` / `20` / `60`            |
| `--volume-ma-window`            | `20`            | `5` / `10` / `20` / `60`            |
| `--rolling-window`              | `252`           | `rolling` 划桶窗口                  |
| `--horizon`                     | `20`            | 前向持有交易日                      |
| `--min-n`                       | `5`             | 参与排名的最小样本数                |
| `--top` / `--bottom`            | `5`             | 最佳 / 最差条数                     |
| `--ternary-q1` / `--ternary-q2` | `0.33` / `0.67` | 三分位下/上界，须 **q1 < q2**       |
| `--json` / `--save-json`        |                 |                                     |

**示例**

```bash
quantlab state-rank --code 515080 --horizon 20 --mode full_sample --top 5 --bottom 5
```

---

## 4. `backtest`

**用途**：单标的回测（分层 tier → 仓位预设；120 日持有、不重叠、次日开盘）。

**参数**

| 参数                                       | 默认            | 说明                                |
| ------------------------------------------ | --------------- | ----------------------------------- |
| `--code` / `--db`                          |                 | 同前                                |
| `--mode`                                   | `rolling`       | 与 signal-research 一致             |
| `--bias-ma`                                | `120`           |                                     |
| `--momentum-window` / `--volume-ma-window` | `10` / `20`     | 同 signal-research                  |
| `--quantile-low` / `--quantile-high`       | `0.33` / `0.67` |                                     |
| `--rolling-window`                         | `252`           |                                     |
| `--bias-q`                                 | 无              | 可选 `Q1` / `Q2-Q4` 等：仅当 **`bias_bucket`** 落在区间内才允许 tier>0 的入场（全样本桶） |
| `--position-rule`                          | `layered`       | `layered` / `conservative` / `full` |
| `--hold-days`                              | `120`           |                                     |
| `--json` / `--save-json`                   |                 |                                     |

**示例**

```bash
quantlab backtest --code 515080 --mode full_sample --bias-ma 120 --position-rule layered --json
```

### 4.1 退出策略（exit）相关参数

`quantlab backtest`（以及 **`quantlab report`**，见下节）在 **`cli/main.py`** 里挂载两套与退出相关的开关：**横评 / 优化**（`_add_exit_rule_cli`）与 **实验与诊断**（`_add_backtest_experiment_cli`）。完整说明以 **`quantlab backtest --help`** 底部 epilog 为准。

**概念（与 help 一致）**

| 层级 | 含义 |
| ---- | ---- |
| **推荐层** | `--signal-preset auto`、`--backtest-preset recommended` 等：用规则层默认 mode、乖离、持有、仓位画像。 |
| **实验层** | `--signal-tier`、`--exit-rule`、`--compare-exit-rules` 等：在**你明确锁定**的假设下做可复现回测；不会用网格最优信号层覆盖你已指定的 `--signal-tier`。 |

**横评与主回测用哪条退出**

| 参数 | 作用 |
| ---- | ---- |
| `--evaluate-exit` | 计算退出规则横评并写入 JSON（`exit_rule_candidates` 等；与分层入场或锁定的 `--signal-tier` 对齐）。 |
| `--optimize-exit` | 在横评里按 **`score_exit_metrics`** 选**满足最小成交**的最优规则，**重算主回测**。与 **`--exit-rule` 互斥**；勿与 **`--compare-profiles`** 同开。 |
| `--exit-rule RULE_ID` | **直接指定**主回测使用的退出规则，与 **`--optimize-exit` 互斥**。可选 id（运行时由 `core.exit_rules.list_cli_exit_rule_ids()` 决定，含 `hold_fixed`）：`hold_fixed`（等价于仅用 `--hold-days` 固定持有）、`time_20`、`time_60`、`time_120`、`state_exit_bottom5`、`state_exit_not_top5`、`momentum_flip_pos`、`bias_flip_pos`、`trend_below_ma20`、`trend_below_ma60`。 |
| `--compare-exit-rules` | 在当前入场设定下横评 **hold_fixed + 全部默认退出**；结果在 JSON 的 **`exit_sweep_under_entry`**（不改变 `optimize` / `exit-rule` 的主线逻辑）。 |

**多目标与入场诊断**

| 参数 | 作用 |
| ---- | ---- |
| `--multi-objective` | 在横评结果上算 Pareto 与各视角最优（JSON：**`multi_objective_decision`**）；**不替代** `score_exit_metrics` 与 **`--optimize-exit`**。 |
| `--objective` | 多目标默认推荐视角：`return_first` \| `risk_first` \| `efficiency_first` \| `robustness_first`（默认 `risk_first`；需横评数据）。 |
| `--entry-diagnostics` | 原始入场 EOD 条件诊断（JSON：**`entry_signal_diagnostics`**）。 |
| `--entry-diagnostics-dates` | 与上项合用：在 JSON 中列出全部 **`raw_entry_dates`**（可能很长）。 |
| `--entry-exit-matching` | 入场 regime 与各退出规则下持仓对齐诊断（JSON：**`entry_exit_matching_diagnostics`**；会隐含入场诊断与退出横评）。 |
| `--entry-exit-top N` | 非 `--json` 时，终端里 ENTRY/EXIT MATCHING 表只打印前 **N** 行（默认全部）。 |

**其它实验参数**

| 参数 | 作用 |
| ---- | ---- |
| `--signal-tier` | 锁定入场层：`NEG` / `NEG_LOW` / `NEG_LOW_HIGH`（仅该层触发）。 |
| `--export-trades PATH` | 主回测成交明细导出 CSV。 |

**示例**

```bash
# 横评 + 按分数优化主回测退出规则
quantlab backtest --code 515080 --evaluate-exit --optimize-exit --json

# 固定使用某条退出规则（与 optimize 二选一）
quantlab backtest --code 515080 --exit-rule time_60 --json

# 全规则对照表 + 多目标
quantlab backtest --code 515080 --compare-exit-rules --multi-objective --objective risk_first --json
```

### 4.2 `recommend` 与退出横评

`quantlab recommend` 提供 **`--include-exit`**：在固定推荐入场下**横评退出规则**，并写入 **`best_exit_rule` / `exit_rule_candidates`**（见该子命令的 `--help`）。

---

## 5. `report`

**用途**：一次跑完 **health + signal-research + backtest**，并写入目录（JSON 包、CSV、Plotly HTML、`SUMMARY.md`）。

**参数**

| 参数                        | 默认      | 说明                                                                              |
| --------------------------- | --------- | --------------------------------------------------------------------------------- |
| 继承 `signal-research` 全套 |           | `--code`、`--db`、`--mode`、`--bias-ma`、三维度、`--rolling-window`、`--horizons` |
| `--position-rule`           | `layered` | 回测仓位                                                                          |
| `--hold-days`               | `120`     |                                                                                   |
| `--output` / `-o`           | **必填**  | 输出目录（会创建）                                                                |
| `--no-json`                 | off       | 跳过 JSON 产物                                                                    |
| `--no-csv`                  | off       | 跳过 CSV                                                                          |
| `--no-charts`               | off       | 跳过 `charts/*.html`                                                              |
| `--print-json`              | off       | 在终端额外打印完整 `ReportResponse` JSON                                          |

**退出策略**：`report` 与 **`backtest` 共用**同一套 exit 相关参数（**`--evaluate-exit`**、**`--optimize-exit`**、**`--exit-rule`**、**`--compare-exit-rules`**、**`--multi-objective`**、**`--objective`**、**`--entry-diagnostics`** 等），见 **§4.1**。

**不包含**：`state-rank`（若需要可后续扩展 `report` 或单独调用）。

**示例**

```bash
quantlab report --code 515080 --mode full_sample --bias-ma 120 --position-rule layered -o ./artifacts/run_001
```

---

## 6. `analyze-path-quality`

**用途**：在 **`from-state`** 当日为起点的样本上，统计 horizon 内是否到达 **`target-state`**（`ever` / `final`），并按 **全样本预计算** 的分位桶汇总 `bias_rate` / `momentum` / `volume_ratio` / `daily_change` 与命中、前向收益。

**与 `--bias-q`**

- 先按 **`bias_bucket` ∈ 指定 Q 区间**（如 `Q3-Q4`）筛起点，再算命中与分解；分解中的 **Q1–Q5** 与筛选使用 **同一套** 全样本桶，不会在子样本上重标。

**常用参数**

| 参数                 | 说明 |
| -------------------- | ---- |
| `--from-state`       | 必填，起点态（前缀匹配，如 `NEG_LOW`） |
| `--target-state`     | 必填，目标态 |
| `--horizon`          | 必填，持有交易日 H |
| `--target-mode`      | `ever`（窗口内任一日达）/ `final`（仅 t+H） |
| `--bucket-features`  | 逗号分隔：`bias_rate,momentum,volume_ratio,daily_change` |
| `--bucket-n`         | **已忽略**（全局固定五分位桶） |
| `--bias-q`           | 可选，如 `Q3-Q4`：仅保留该乖离桶内的起点 |
| `--mode` / 三维度窗  | 与 `state-rank` 研究帧一致（`--bias-ma` 等） |

**示例**

```bash
quantlab analyze-path-quality \
  --code 515080 --db ./etf_data.db \
  --from-state NEG_LOW --target-state POS_HIGH_HIGH \
  --horizon 60 --bias-q Q3-Q4 --json
```

---

## 7. `analyze-path-rules`

**用途**：在与 path-quality 相同的起点 / 目标 / horizon 设定下，挖掘单因子或双因子的 **连续 Q 区间** 规则（如 `bias_rate in Q2-Q4`）；桶标签为 **全局 Q1–Q5**。

**常用参数**

| 参数                         | 说明 |
| ---------------------------- | ---- |
| `--features`                 | 逗号分隔特征名（默认 `bias_rate,volume_ratio`） |
| `--max-combinations`         | `1` 仅单因子；`2` 含双因子 AND |
| `--min-count` / `--top-k`    | 最小样本数、返回规则条数上限 |
| `--rules-above-baseline-only`| 仅保留 hit_rate ≥ 基线的规则 |
| `--bucket-n`                 | **已忽略** |
| `--bias-q`                   | 可选，先按 `bias_bucket` 筛起点 |

---

## 8. `analyze-transition`

**用途**：给定 **`--from-state`**，统计各 **`--horizons`** 上未来态出现频次（研究帧，无交易）。

**示例**

```bash
quantlab analyze-transition --code 515080 --from-state NEG_LOW --horizons 5,10,20 --json
```

---

## 9. `recommend`

**用途**：按内置规则给出标的适配度与默认信号参数（不执行事件研究）。详见 `quantlab recommend --help`。

**退出**：可选 **`--include-exit`** 在推荐入场下横评退出规则（见 **§4.2**）。

---

## JSON 与 Schema

- 结构体定义：`core/schemas.py`（Pydantic）。
- `--json` 输出为 `model_dump_json()`，字段与 Streamlit/API 一致，便于自动化 review。
- `signal-research` / `backtest` 中 **图表** 为 `ChartSpec.plotly_json`（由 Plotly `Figure.to_json()` 反序列化友好结构）。

---

## 与代码的对应关系

```
cli/main.py          →  argparse，组装 *Request，调用 core.runner
core/runner.py       →  run_health | run_signal_research | run_recommendation | run_state_ranking |
                       run_state_transition | run_path_quality | run_path_rule_mining |
                       run_backtest | run_report
core/pipeline.py     →  prepare_research_frame（含全样本 *_bucket 列）
core/schemas.py      →  请求/响应模型
```

修改行为时应改 **`core`**，再视需要同步本 README 与 `cli/main.py` 的 help 字符串。
