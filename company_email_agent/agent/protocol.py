"""A one-call text protocol for the course's plain chat interface."""

import json
import re

from company_email_agent.types import ParsedResponse, ToolCall


_OPEN = re.compile(
    r"^[ \t]*(?:```tool_call|(?P<dsml><｜｜DSML｜｜[ \t]*tool_call>))[ \t]*(?:\r?\n|$)", re.MULTILINE,
)
_CLOSE = re.compile(r"[ \t]*\r?\n[ \t]*```[ \t]*(?=\r?\n|$)")
_DSML_CLOSE = re.compile(r"[ \t]*\r?\n[ \t]*</｜｜DSML｜｜[ \t]*tool_call>[ \t]*(?=\r?\n|$)")
_DSML_TAG = re.compile(r"</?｜｜DSML｜｜")
_DSML_CALLS = re.compile(
    r'<｜｜DSML｜｜ calls>[ \t]*\r?\n'
    r'<｜｜DSML｜｜ invoke name="(?P<name>[A-Za-z_][A-Za-z0-9_-]*)">\s*'
    r'(?P<parameters>.*?)'
    r'</｜｜DSML｜｜ invoke>\s*</｜｜DSML｜｜ calls>',
    re.DOTALL,
)
_DSML_PARAMETER = re.compile(r'<｜｜DSML｜｜ parameter name="([A-Za-z_][A-Za-z0-9_-]*)" string="(true|false)">')
_DSML_PARAMETER_END = "</｜｜DSML｜｜ parameter>"
_UNSUPPORTED_CALL = "Unsupported tool-call format. Reissue at most one call in a fenced tool_call block with JSON."


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate JSON keys")
    return fields


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


_DECODER = json.JSONDecoder(object_pairs_hook=_unique_fields, parse_constant=_invalid_constant)


def render_tools(schemas: list[dict]) -> str:
    tools = "\n".join(json.dumps(schema, separators=(",", ":")) for schema in schemas)
    return f"""Available tools, one JSON schema per line:
{tools}

To use a tool, write exactly one fenced tool_call block on its own lines:
```tool_call
{{"name":"tool_name","args":{{"parameter":"value"}}}}
```
Replace the example name and arguments with a listed tool and schema. Use valid JSON,
including double-quoted keys, the required argument names, and the correct value types.
Use an empty args object for a tool with no arguments. If a tool is needed, make the
call now and wait for its result before proceeding. When you have the final answer,
return ordinary text with no tool_call block.
"""


def format_tool_call(text: str, call: ToolCall) -> str:
    payload = json.dumps({"name": call.name, "args": call.args}, allow_nan=False)
    block = f"```tool_call\n{payload}\n```"
    return f"{text.strip()}\n\n{block}" if text.strip() else block


def _native_dsml(text: str) -> ToolCall | None:
    match = _DSML_CALLS.fullmatch(text)
    if match is None:
        return None
    parameters = match["parameters"]
    args = {}
    pos = 0
    try:
        while pos < len(parameters):
            field = _DSML_PARAMETER.match(parameters, pos)
            if field is None or field[1] in args:
                return None
            end = parameters.find(_DSML_PARAMETER_END, field.end())
            if end < 0:
                return None
            value = parameters[field.end():end]
            if _DSML_TAG.search(value):
                return None
            args[field[1]] = value if field[2] == "true" else _DECODER.decode(value)
            pos = end + len(_DSML_PARAMETER_END)
            while pos < len(parameters) and parameters[pos].isspace():
                pos += 1
    except (ValueError, RecursionError):
        return None
    if args.keys() == {"args"} and isinstance(args["args"], dict):
        args = args["args"]
    return ToolCall(match["name"], args)


def parse_response(text: str) -> ParsedResponse:
    marker = _OPEN.search(text)
    if marker is None:
        plain = text.strip()
        if _DSML_TAG.search(plain):
            call = _native_dsml(plain)
            return ParsedResponse("", call) if call is not None else ParsedResponse(plain, error=_UNSUPPORTED_CALL)
        return ParsedResponse(plain)

    # Decode first, so marker-looking text inside JSON strings is just data.
    start = marker.end()
    while start < len(text) and text[start] in " \t\r\n":
        start += 1
    try:
        payload, end = _DECODER.raw_decode(text, start)
    except (ValueError, RecursionError):
        return ParsedResponse(text.strip(), error="Malformed or incomplete tool-call JSON.")
    terminator = _DSML_CLOSE if marker.group("dsml") else _CLOSE
    closing = terminator.match(text, end)
    if closing is None:
        return ParsedResponse(text.strip(), error="Close the tool_call block after its JSON object.")
    if _OPEN.search(text, closing.end()) is not None:
        return ParsedResponse(text.strip(), error="Only one tool call is allowed per reply.")
    # Unrecognized DSML around a valid block must not allow partial execution.
    if _DSML_TAG.search(text, 0, marker.start()) or _DSML_TAG.search(text, closing.end()):
        return ParsedResponse(text.strip(), error=_UNSUPPORTED_CALL)
    if (not isinstance(payload, dict) or payload.keys() != {"name", "args"}
            or not isinstance(payload["name"], str) or not isinstance(payload["args"], dict)):
        return ParsedResponse(text.strip(), error="A call needs a tool name string and an args object.")

    plain = (text[:marker.start()] + text[closing.end():]).strip()
    return ParsedResponse(plain, ToolCall(payload["name"], payload["args"]))
