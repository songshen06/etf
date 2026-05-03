# `etf`（面向 Agent 的 `quantlab` CLI + SQLite 数据库）

这个仓库的目标是让 AI Agent 以**稳定、可脚本化、可解析**的方式调用 `quantlab` 做 ETF 研究与策略决策。

- 默认自带 SQLite：`./etf_data.db`
- 默认入口命令：`quantlab <subcommand> ...`（安装后）
- 不想安装也能跑：`python -m cli.main ...`

更完整的命令速查表见：[cli/README.md](./cli/README.md)

## 快速开始（本地）

在仓库根目录：

```bash
pip install -e .
quantlab --help
quantlab <subcommand> --help
```

不安装包：

```bash
python -m cli.main --help
python -m cli.main <subcommand> --help
```

## 数据库路径规则（Agent 需要知道）

多数命令会读 SQLite。解析顺序：

1) 显式参数（例如 `--db ./etf_data.db`）
2) 环境变量 `ETF_DB_PATH`
3) 仓库根目录 `./etf_data.db`

Agent 编排时建议始终显式传 `--db`，保证可复现。

## 推荐的 Agent 调用姿势

### 1) 优先请求 JSON 输出

很多子命令支持 `--json`（结构化输出），Agent 解析更稳：

```bash
quantlab detect-market-regime --db ./etf_data.db --json
quantlab recommend-portfolio --db ./etf_data.db --json
quantlab score-bloody-chip --code 510300 --db ./etf_data.db --json
quantlab recommend-strategy --etf 159361 --db ./etf_data.db --current-position 0.0 --json
```

### 2) 用 subprocess 的 argv 列表（避免引号坑）

```python
import subprocess

cp = subprocess.run(
    ["quantlab", "recommend-strategy", "--etf", "159361", "--db", "./etf_data.db", "--current-position", "0.0", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
print(cp.stdout)
```

### 3) 注意有状态命令

`dividend-signal` 可能写入 `~/.quantlab/dividend_state.json`。如果你只是做“建议预览”，优先使用 `--dry-run`。

## 常用工作流（面向 Agent）

### A. 三 ETF 策略快检（A500 / 红利质量 / 中证红利）

```bash
quantlab recommend-strategy --etf 159361 --db ./etf_data.db --current-position 0.0 --json
quantlab recommend-strategy --etf 159209 --db ./etf_data.db --current-position 0.0 --json
quantlab recommend-strategy --etf 515080 --db ./etf_data.db --current-position 0.0 --json
```

### B. 市场状态 → 组合推荐 → 回测 → 实盘调仓建议

```bash
quantlab detect-market-regime --db ./etf_data.db --json
quantlab recommend-portfolio --db ./etf_data.db --json
quantlab backtest-portfolio-regime --db ./etf_data.db --start-date 2020-02-06 --rebalance monthly --json
quantlab recommend-portfolio-action --db ./etf_data.db --current-a500 0.4 --current-dividend 0.3 --current-dividend-growth 0.3 --json
```

### C. 数据质量体检

```bash
quantlab health --code 510300 --db ./etf_data.db --json
```

### D. 血筹评分（单只 / 排名）

```bash
quantlab score-bloody-chip --code 510300 --db ./etf_data.db --json
quantlab rank-bloody-chip --db ./etf_data.db --top-k 10
```

血筹默认配置文件：`./configs/bloody_chip_etf.yaml`

## 更新数据（回填 / 增量）

仓库根目录提供 `update_db.py` 用于更新 `etf_data.db`：

```bash
python update_db.py --db-path ./etf_data.db --smart --backfill-missing
```

依赖建议：

- 只跑 CLI：`pip install -e .`
- 需要抓取/更新数据（akshare/requests）：`pip install -e '.[all]'`

## 目录结构（与 Agent 相关）

- `cli/`：`quantlab` 主入口（参数解析 + 调用核心逻辑）
- `core/`：研究帧与指标计算的核心实现
- `quantlab/`：扩展子命令（血筹、红利信号、策略系统、regime 组合等）
- `configs/`：运行时配置（例如血筹配置）
- `etf_data.db`：SQLite 数据库（默认数据源）
- `update_db.py`：更新数据库脚本

