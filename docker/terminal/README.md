# Company Email Agent Docker 终端

受管终端使用 Windows Docker Desktop 的本地 Linux Engine。每次命令在独立容器中执行，同一会话共享限额工作卷。根文件系统和命令容器的语料卷只读，网络禁用，容器使用非 root 用户并限制 CPU、内存、进程数、执行时间和输出。

## 本地配置

启动 Docker Desktop 的 Linux 容器模式。在仓库根目录打开 PowerShell。以下命令构建镜像并把本机 Engine ID 与镜像 ID 写入被忽略的配置文件。

```powershell
$docker = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
$engine = 'npipe:////./pipe/dockerDesktopLinuxEngine'
$image = 'company-email-agent-terminal'
& $docker --host $engine build --tag $image docker/terminal
if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed' }

$engineId = & $docker --host $engine info --format '{{.ID}}'
if ($LASTEXITCODE -ne 0) { throw 'Could not verify the local Linux Engine' }
$imageId = & $docker --host $engine image inspect --format '{{.Id}}' $image
if ($LASTEXITCODE -ne 0) { throw 'Could not verify the local terminal image' }

New-Item -ItemType Directory -Force -Path .lm_cache | Out-Null
@{ engine_id = $engineId; image_id = $imageId } |
    ConvertTo-Json -Compress |
    Set-Content -Encoding Ascii .lm_cache/docker-terminal.json
```

将自行取得的、与 `data/source.json` 匹配的课程压缩包放在本地 `data/emails.jsonl.gz`。命令容器从压缩包校验并加载语料。邮件检索还需通过 `data/download.py` 安装解压文件。

执行器验证 Engine、镜像、权限和挂载。缺少有效配置或校验失败时返回工具错误。Docker 配置、课程压缩包及会话工作卷属于本地运行环境。

## 固定命令探针

完成上述配置后，可在同一 PowerShell 中显式启用无害实机探针。

```powershell
$env:CARDINAL_DOCKER_PROBES = '1'
$env:CARDINAL_PROBE_ENGINE = $engineId
$env:CARDINAL_PROBE_IMAGE = $imageId
uv run pytest tests/test_docker_terminal_security.py -m "docker and not live"
```

探针验证当前配置的隔离属性和清理行为。基础镜像使用固定摘要的 Docker Official Alpine，各系统软件包保留上游归属。 [Dockerfile](Dockerfile) · [容器入口](entrypoint.py)
