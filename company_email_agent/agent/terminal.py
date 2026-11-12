"""Format a result from the session's isolated Docker terminal."""

from company_email_agent.agent.context import clip_middle
from company_email_agent.sandbox_docker import DockerTerminal
from company_email_agent.tokens import CHARS_PER_TOKEN
from company_email_agent.types import AgentConfig


def terminal_output(cmd: str, config: AgentConfig, terminal: DockerTerminal) -> str:
    limit = min(7200, max(128, config.context_budget * CHARS_PER_TOKEN // 3))
    result = terminal.run(cmd, timeout=config.terminal_timeout, preview=limit)
    status = f"Exit code: {result.exit_code if result.exit_code is not None else 'unavailable'}"
    if result.reason:
        status += f"\n{result.reason}"

    stdout, stderr = result.stdout, result.stderr
    if len(stdout) + len(stderr) > limit:
        status += "\nOutput preview."
        if result.output_file:
            status += f" Captured output saved to {result.output_file}."
        if result.stderr_file:
            status += f" Captured stderr saved to {result.stderr_file}."
        if not result.output_file and not result.stderr_file:
            status += " The session could not save the captured output."

        stderr_room = min(len(stderr), max(limit // 3, limit - len(stdout)))
        stdout = clip_middle(stdout, limit - stderr_room)
        stderr = clip_middle(stderr, stderr_room)

    content = f"{status}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    if result.rejected or result.timed_out or result.exit_code != 0:
        raise RuntimeError(content)
    return content
