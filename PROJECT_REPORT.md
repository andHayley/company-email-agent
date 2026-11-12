# Company Email Agent 项目报告

Company Email Agent 将邮件索引、文档检索和模型工具调用接入同一个会话。工具返回证据，Agent 根据证据决定继续检索、请求澄清或给出答案。

## 实现

| 模块 | 设计 |
| --- | --- |
| AgentSession | 管理工具调用、上下文压缩、回合预算和工具结果。DeepSeek 与 OpenRouter 共用模型接口。 |
| 邮件检索与处理 | BM25 对主题和正文建索引。Email QA 利用首轮证据生成后续查询。Priority 和 Daily Digest 共享邮件分类能力，摘要按上下文预算组装。 |
| 记忆与审批 | 记忆按会话配置写入本地目录。工具先校验参数和拒绝规则，确认模式下的非只读工具还会请求审批。 |
| Docker 终端 | 每条命令使用独立容器，同会话共享限额工作卷。语料只读，网络禁用。启动或隔离验证失败时拒绝执行。 |

[Agent 会话](company_email_agent/agent/session.py) · [邮件流水线](company_email_agent/pipelines/) · [Docker 终端](docker/terminal/README.md)

## 原项目评估记录

以下数字来自发布整理前、具备课程语料和 Docker 配置的项目验收。使用 `deepseek-flash` 的 20 场景评估从 12/20 提升到 18/20，达到 16/20 门槛，该评估项得分 10/10。复测中原先通过的 12 项全部保留，6 个失败项恢复。指定 Windows Docker Desktop 环境通过了 7/7 项隔离探针。

复测共记录 662 条 API 响应，输入 825,782 Token，输出 31,485 Token，项目用量账本估算 $0.105245502。

- 编号更正场景仍可能召回新旧两个值。模型将值保存为自然句时，现有标签去重没有覆盖该格式。
- 邮件多跳场景曾在一轮收到多工具调用和未支持的 DSML 变体后进入 `malformed`。后续答对仍触发原程序检查。
- 原项目完整离线回归有 395 项通过，既有 BM25 性能断言实测 17.5 秒，门槛为 15 秒。

发布副本只保留汇总。原始邮件、对话、工具结果和本机运行记录保存在原项目中。

## 发布副本验证

包导入、三个 CLI 帮助入口、锁文件检查和 wheel 构建已通过。发布准备的最终默认离线回归在无 API 密钥、邮件归档、Docker 本机配置或历史夹具的条件下运行，结果为 **381 通过、0 失败、0 跳过、0 错误、65 未选中**。

- DSML 与 Memory 定向回归 46 项通过。公开夹具使用新编的协议输入。18 项私有历史回复重放只做收集检查。
- 长输出回归用模拟执行器验证两种预算下的真实 Agent 工具格式和上下文交接。3 项通过，包含拒绝只出现在助手命令中的标记。两项真实 Docker 用例已收集，未执行。
- 模拟评估模块 8 项离线通过。发布副本实际 CLI 在发现只有 18 个角色时返回退出码 2，并列出缺失角色。显式子集会显示实际数量和按原规则计算的门槛。

归档、实机 Docker 和 Live 用例只做筛选与收集检查，未计入通过结果。

## 本地配置与测试

默认离线测试运行 `uv run pytest`。安装依赖、试跑文档检索和启动会话的步骤见 [快速开始](README.md#快速开始)。

现有 `CS329Z_*` 环境变量可设置模型、缓存和预算。聊天记录写入 `runs/`。模型缓存、用量账本和默认记忆保存在 `.lm_cache/`。可运行 `uv run python -m company_email_agent.llm` 查看本地用量。

| 范围 | 运行命令 |
| --- | --- |
| 仅检查默认收集 | `uv run pytest --collect-only` |
| 邮件归档离线测试 | `uv run pytest -m "archive and not docker and not live"` |
| Docker 离线集成 | `uv run pytest -m "docker and not live"` |
| 私有历史回复重放 | `uv run pytest -m "historical_fixture and not live"` |
| Live 测试 | `uv run pytest -m "live"` |
| 同时依赖语料和 Docker 的 Live 测试 | `uv run pytest -m "live and docker and archive"` |

`archive`、`docker`、`historical_fixture` 和 `live` 可以组合。为离线集成写自定义筛选时，请保留 `not live`。向任一命令加 `--collect-only` 可预览选中的用例。

| 本地资源 | 适用范围 |
| --- | --- |
| 解压的 `data/emails/emails.jsonl` | 归档检索、摘要等测试 |
| 校验过的 `data/emails.jsonl.gz` 和本机 Docker 固定配置 | 实际执行命令的集成测试 |
| `tests/fixtures/historical/dsml_memory_trace.json`、`tests/fixtures/historical/memory_alias_live_trace.json` | 可选的原始回复重放。将已有原件放在本地忽略目录。默认测试用各自的公开合成夹具。 |
| 本地 `priority_gold.json`、`terminal_questions.json` | Priority Live 和 Terminal Live 的原题集 |
| `09-coding-no-python.json`、`20-coding-big-listing.json` | 完整评估缺少的两个原始角色。副本含其余 18 个角色。 |

模拟评估默认预检原始 20 角色名单，缺失时在请求模型前停止。可显式使用 `--slice` 或 `--only` 运行子集，结果会显示实际角色数及按原规则计算的门槛。 [模拟评估说明](company_email_agent/simulation/README.md)

Live 测试需要用户自行配置模型密钥并会产生 API 用量。Docker 实机安全探针需要另外启用 `CARDINAL_DOCKER_PROBES`。 [Docker 配置](docker/terminal/README.md)

## 归属

项目源自 Stanford [CS329Z Engineering AI Agents 的 Assignment 1 Harness](https://github.com/cs329z/assignment1-harness)。课程框架由 Michael Ryan 设计。感谢 Houjun (Jack) Liu、Owen Queen、Shreyas Sharma 和 Anantharaman Iyer 试做并反馈。保留原有 [MIT License](LICENSE) 和版权声明。Docker 终端的 Alpine 基础镜像和系统软件包保留各自的上游归属。
