# Company Email Agent

A personal AI agent for importing, searching, understanding, and managing enterprise email data.

在一个会话里检索本地邮件和公司文档、追查跨邮件线索，并生成带出处的答案。

## 核心功能

- **邮件检索与问答** · 用 BM25 查找相关邮件，沿已找到的线索补充检索并引用证据。
- **优先级与每日摘要** · 给邮件分级，汇总当天的紧急事项和常规更新。
- **文档与跨会话记忆** · 按文档和行号查看出处，记住用户定义的简称和长期偏好。
- **受控工具执行** · 工具调用经过参数校验、拒绝规则和审批。Docker 终端在独立容器中执行命令。

## 验证结果

| 验证 | 结果 | 范围 |
| --- | --- | --- |
| 默认离线测试 | **381 项通过** | 无 API 密钥、邮件归档、Docker 或私有夹具 |
| 端到端模拟评估 | **18/20 场景通过** | 原项目完整题集，使用 `deepseek-flash` 复测，达到 16/20 门槛 |

评估设置、剩余错例、用量和完整测试命令见 [项目报告](PROJECT_REPORT.md)。

## 快速开始

需要 Python 3.11+ 和 uv。在仓库根目录运行以下命令，即可用随仓库提供的示例文档试跑检索和默认离线测试。

```sh
uv sync --locked
uv run python -m company_email_agent.show search_docs "parental leave"
uv run pytest
```

### 使用本地邮件

自行准备与 `data/source.json` 匹配的课程邮件压缩包。安装脚本会校验并解压归档。

```sh
uv run python data/download.py --from-file "path/to/emails.jsonl.gz"
uv run python -m company_email_agent.show bm25 "larchfield audit"
```

### 启动交互会话

将 `.env.example` 复制为本地 `.env`。使用 OpenRouter 时填写 `OPENROUTER_API_KEY`。使用 DeepSeek 时设置 `LLM_PROVIDER=deepseek` 并填写 `DEEPSEEK_API_KEY`。

```sh
uv run python -m company_email_agent.chat --mode confirm
```

使用在线模型时，相关问题、邮件、文档或工具结果可能发送给所选服务商。 [Docker 终端配置](docker/terminal/README.md) · [模拟评估说明](company_email_agent/simulation/README.md)
