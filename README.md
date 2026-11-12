# Company Email Agent

A personal AI agent for importing, searching, understanding, and managing enterprise email data.

Company Email Agent 可检索本地邮件归档、追查跨邮件线索、分类邮件并生成摘要。会话支持文档检索、跨会话记忆、工具审批和受控的 Docker 终端。

[项目设计与评估](PROJECT_REPORT.md) · [Docker 终端配置](docker/terminal/README.md) · [模拟评估说明](company_email_agent/simulation/README.md)

## 安装与本地试用

需要 Python 3.11+ 和 uv。在仓库根目录运行。

```sh
uv sync --locked
uv run python -m company_email_agent.chat --help
uv run python -m company_email_agent.show search_docs "parental leave"
```

仓库带有 Cardinal Energy 的课程示例文档。邮件检索和邮件处理需要使用者自行准备匹配的课程归档。取得有权使用的压缩包后，运行以下命令进行完整性校验和解压。

```sh
uv run python data/download.py --from-file /path/to/emails.jsonl.gz
uv run python -m company_email_agent.show bm25 "larchfield audit"
```

安装脚本会写入 `data/emails/emails.jsonl`，且可按 `data/source.json` 所列地址尝试获取压缩包。Docker 终端还需要本地的 `data/emails.jsonl.gz`，具体步骤见 Docker 配置。

## 模型与会话

将 `.env.example` 复制为本地 `.env`。默认服务商为 OpenRouter，使用 `OPENROUTER_API_KEY`。DeepSeek 用户可设置 `LLM_PROVIDER=deepseek` 和 `DEEPSEEK_API_KEY`。现有 `CS329Z_*` 环境变量仍可设置模型、缓存和预算。

```sh
uv run python -m company_email_agent.chat --mode confirm
uv run python -m company_email_agent.llm
```

聊天记录写入 `runs/`，模型缓存、用量账本和默认记忆保存在 `.lm_cache/`。启用真实模型后，当前问题及相关邮件、文档或工具结果可能发送给所选服务商。

## 测试

安装锁定依赖后，直接运行默认离线套件。该套件使用脚本模型、合成夹具和随仓库提供的示例文档。

```sh
uv run pytest
```

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
