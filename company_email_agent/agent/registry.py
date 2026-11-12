"""Tool registration, JSON Schema validation, and contained execution."""

from copy import deepcopy

from jsonschema import Draft202012Validator

from company_email_agent.types import ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]):
        self._tools = {tool.name: tool for tool in tools}
        self._validators = {
            name: Draft202012Validator(deepcopy(tool.parameters))
            for name, tool in self._tools.items()
        }

    def schemas(self) -> list[dict]:
        return [
            {"name": name, "description": tool.description,
             "parameters": deepcopy(self._validators[name].schema)}
            for name, tool in self._tools.items()
        ]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        if not isinstance(name, str) or name not in self._tools:
            return ToolResult("unknown_tool", f"Unknown tool: {name!r}.")
        if not isinstance(args, dict):
            return ToolResult("invalid_args", f"{name} requires a JSON object for arguments.")
        try:
            error = next(self._validators[name].iter_errors(args), None)
            if error is not None:
                return ToolResult("invalid_args", f"Invalid arguments for {name}: {error.message}")
        except Exception as exc:
            return ToolResult("invalid_args", f"Cannot validate arguments for {name}: {exc}")
        return None

    def is_read_only(self, name: str) -> bool:
        return self._tools[name].read_only

    def execute(self, name: str, args: dict) -> ToolResult:
        error = self.validate(name, args)
        return error if error is not None else self.invoke(name, args)

    def invoke(self, name: str, args: dict) -> ToolResult:
        """Execute a call already validated by the session."""
        try:
            return ToolResult("ok", str(self._tools[name].fn(**args)))
        except Exception as exc:
            return ToolResult("tool_error", f"{name}: {exc or type(exc).__name__}")
