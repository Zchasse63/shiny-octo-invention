# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Command-line interface for VectorBT PRO.

This module exposes a Typer-based CLI for interacting with VectorBT PRO from the shell, including:

* high-level chat commands such as `vbt chat`
* the MCP server entry point under `vbt mcp serve`
* direct invocation of all tools registered in `vectorbtpro.mcp.tool_registry`

The CLI is intended to be friendly both for humans and for AI agents operating in a terminal environment.

## Entry points

After installing the package with its console script, the CLI can be invoked as:

```bash
vbt ...
```

The same interface is also available via the module entry point:

```bash
python -m vectorbtpro ...
```

For example:

```bash
vbt chat "What is PFO?"
python -m vectorbtpro interact "List attributes of PFO"
vbt mcp serve
python -m vectorbtpro mcp find "PFO"
```

## Command layout

Top-level commands:

* `vbt chat`
* `vbt quick-chat`
* `vbt interact`
* `vbt mcp`

The `mcp` group contains:

* `vbt mcp serve` to run the MCP server itself
* `vbt mcp <tool-name>` to invoke any tool from `tool_registry`

Tool names are exposed as hyphenated command names. For example:

* `run_code` becomes `vbt mcp run-code`

## Discovering commands and tools

List top-level commands:

```bash
vbt --help
```

List all MCP-related commands, including all registered tools:

```bash
vbt mcp --help
```

Get help for a specific command or tool:

```bash
vbt chat --help
vbt mcp find --help
```

The help text for tool commands is derived from the underlying function docstrings and preserves
paragraph breaks to make long API-style documentation more readable in the terminal.

## Flexible invocation with `--call`

Many commands also expose a `--call` option intended as an escape hatch for agents and advanced users.
It accepts JSON or a Python literal and can supply arbitrary positional and keyword arguments
to the underlying function.

Supported shapes:

* `{"args": [...], "kwargs": {...}}`
* `{...}` interpreted as keyword arguments
* `[...]` or `(...)` interpreted as positional arguments

Examples:

```bash
vbt chat --call '{"args": ["What is PFO?"], "kwargs": {"rank_kwargs": {"rerank": true}}}'
vbt mcp run-code --call '{"kwargs": {"code": "print(vbt.__version__)"}}'
```

Explicitly passed CLI options override matching values from `--call`.
"""

import ast
import functools
import inspect
import json
import sys
import typing

import click
import typer

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.mcp import tool_registry
from vectorbtpro.mcp_server import main as mcp_main

__all__ = []


@define
class CLICommand(DefineMixin):
    """Metadata describing a registered CLI command."""

    func: tp.Callable = define.field()
    """Function invoked by the CLI command."""

    name: str = define.field()
    """Public command name."""

    help: tp.Optional[tp.Any] = define.field(default=None)
    """Help text displayed by Typer, or a callable returning it."""

    kwargs: tp.Dict[str, tp.Any] = define.field(factory=dict)
    """Extra keyword arguments passed to `Typer.command`."""


cli_registry = []
"""Registry of explicitly declared CLI commands."""

CALL_HELP = (
    "Extra call payload as JSON or Python literal. "
    "Supports {'args': [...], 'kwargs': {...}}, a kwargs dict, or a positional args list."
)


def cli_command(
    arg: tp.Union[None, str, tp.Callable] = None,
    /,
    *,
    name: tp.Optional[str] = None,
    help: tp.Optional[tp.Any] = None,
    **kwargs,
) -> tp.Callable:
    """Decorate and register a function as a Typer CLI command.

    Args:
        arg (Union[None, str, Callable]): Function to decorate or command name.
        name (Optional[str]): Explicit command name.
        help (Optional[Any]): Command help text or a callable returning it.
        **kwargs: Extra keyword arguments passed to `Typer.command`.

    Returns:
        Callable: Decorator or the decorated function.
    """
    if isinstance(arg, str) and name is None:
        name = arg
        arg = None

    def wrapper(func: tp.Callable) -> tp.Callable:
        command_name = name or func.__name__.replace("_", "-")
        if help is None:
            command_help = format_cli_help(inspect.getdoc(func))
        else:
            command_help = help
        cli_registry.append(CLICommand(func=func, name=command_name, help=command_help, kwargs=kwargs))
        return func

    if callable(arg):
        return wrapper(arg)
    return wrapper


def format_cli_help(help_text: tp.Optional[str]) -> tp.Optional[str]:
    """Format help text for Click/Typer without collapsing paragraph newlines.

    Click preserves line breaks for paragraphs that begin with the backspace
    marker `\b`. This function adds that marker to each paragraph separated by
    blank lines so docstrings keep their manual formatting in CLI help.

    Args:
        help_text (Optional[str]): Raw help text or docstring.

    Returns:
        Optional[str]: Help text formatted for Click/Typer.
    """
    if help_text is None:
        return None
    help_text = inspect.cleandoc(help_text).strip()
    if len(help_text) == 0:
        return help_text
    paragraphs = help_text.split("\n\n")
    return "\n\n".join("\b\n" + paragraph for paragraph in paragraphs)


def resolve_cli_help(help_value: tp.Optional[tp.Any]) -> tp.Optional[str]:
    """Resolve and format CLI help text.

    Args:
        help_value (Optional[Any]): Help text or a callable returning help text.

    Returns:
        Optional[str]: Help text formatted for Click/Typer.
    """
    if callable(help_value):
        help_value = help_value()
    return format_cli_help(help_value)


def parse_call_payload(call: tp.Optional[str]) -> tp.Tuple[tp.List[tp.Any], tp.Dict[str, tp.Any]]:
    """Parse a structured `--call` payload.

    The payload can be JSON or a Python literal. Supported shapes are:

    * `{"args": [...], "kwargs": {...}}`
    * `{...}` interpreted as keyword arguments
    * `[...]` or `(...)` interpreted as positional arguments

    Args:
        call (Optional[str]): Raw payload string.

    Returns:
        Tuple[List[Any], Dict[str, Any]]: Parsed positional and keyword arguments.
    """
    if call is None:
        return [], {}
    parsed = None
    errors = []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(call)
            break
        except Exception as exc:
            errors.append(exc)
    if parsed is None:
        raise ValueError(f"Could not parse --call payload: {errors[-1]}")
    if isinstance(parsed, dict):
        keys = set(parsed.keys())
        special_keys = {"args", "kwargs"}
        if keys & special_keys:
            if not keys <= special_keys:
                raise TypeError("When using special --call keys, only 'args' and/or 'kwargs' are allowed")
            args = parsed.get("args", [])
            kwargs = parsed.get("kwargs", {})
        else:
            args = []
            kwargs = parsed
    elif isinstance(parsed, (list, tuple)):
        args = list(parsed)
        kwargs = {}
    else:
        args = [parsed]
        kwargs = {}
    if not isinstance(args, (list, tuple)):
        raise TypeError("'args' in --call payload must be a list or tuple")
    if not isinstance(kwargs, dict):
        raise TypeError("'kwargs' in --call payload must be a dict")
    return list(args), dict(kwargs)


def get_cli_overrides(ctx: click.Context, exclude: tp.Optional[tp.Set[str]] = None) -> tp.Dict[str, tp.Any]:
    """Return parameters supplied explicitly on the command line.

    Args:
        ctx (click.Context): Current Click context.
        exclude (Optional[Set[str]]): Parameter names to ignore.

    Returns:
        Dict[str, Any]: Mapping of explicitly passed parameter names to values.
    """
    if exclude is None:
        exclude = set()
    overrides = {}
    for name, value in ctx.params.items():
        if name in exclude:
            continue
        source = ctx.get_parameter_source(name)
        if source is click.core.ParameterSource.COMMANDLINE:
            overrides[name] = value
    return overrides


def is_list_like_annotation(annotation: tp.Any) -> bool:
    """Check if a type annotation represents a list-like structure.

    Args:
        annotation (Any): Type annotation to check.

    Returns:
        bool: True if the annotation is list-like, False otherwise.
    """
    origin = typing.get_origin(annotation)
    if origin in (list, tp.List):
        return True
    if origin is typing.Union:
        return any(is_list_like_annotation(arg) for arg in typing.get_args(annotation) if arg is not type(None))
    return False


def merge_call_arguments(
    func: tp.Callable,
    call: tp.Optional[str] = None,
    cli_overrides: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> tp.Tuple[tp.List[tp.Any], tp.Dict[str, tp.Any]]:
    """Merge a `--call` payload with explicit CLI arguments.

    The payload acts as a flexible base call specification, while explicitly passed
    CLI arguments override matching parameters.

    Args:
        func (Callable): Target callable.
        call (Optional[str]): Structured payload string.
        cli_overrides (Optional[Dict[str, Any]]): Explicit CLI arguments.

    Returns:
        Tuple[List[Any], Dict[str, Any]]: Positional and keyword arguments ready to call.
    """
    signature = inspect.signature(func)
    call_args, call_kwargs = parse_call_payload(call)
    bound = signature.bind_partial(*call_args, **call_kwargs)

    for name, value in (cli_overrides or {}).items():
        param = signature.parameters.get(name)
        if param is None:
            continue
        if isinstance(value, tuple) and is_list_like_annotation(param.annotation):
            value = list(value)
        bound.arguments[name] = value

    args = []
    kwargs = {}
    for param in signature.parameters.values():
        if param.name not in bound.arguments:
            continue
        value = bound.arguments[param.name]
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            args.extend(value)
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs.update(value)
        else:
            kwargs[param.name] = value
    return args, kwargs


def validate_required_arguments(
    func: tp.Callable,
    args: tp.Sequence[tp.Any],
    kwargs: tp.Dict[str, tp.Any],
    ctx: click.Context,
) -> None:
    """Validate that all required function parameters are present after CLI/`--call` merging.

    Args:
        func (Callable): Target callable whose required parameters should be checked.
        args (Sequence[Any]): Positional arguments after merging `--call` and CLI overrides.
        kwargs (Dict[str, Any]): Keyword arguments after merging `--call` and CLI overrides.
        ctx (click.Context): Current Click context used to attach usage errors to the command.

    Returns:
        None
    """
    signature = inspect.signature(func)
    bound = signature.bind_partial(*args, **kwargs)

    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        if param.name in bound.arguments:
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            raise click.UsageError(f"Missing option '--{param.name.replace('_', '-')}'", ctx=ctx)
        raise click.UsageError(f"Missing argument '{param.name}'", ctx=ctx)


def print_output(output: tp.Any) -> None:
    """Print CLI output to standard output.

    Args:
        output (Any): Output value to print.

    Returns:
        None
    """
    if output is None:
        return
    output = str(output)
    if len(output) == 0:
        return
    sys.stdout.write(output)
    if not output.endswith("\n"):
        sys.stdout.write("\n")


def build_tool_signature(func: tp.Callable) -> inspect.Signature:
    """Build a Typer-safe signature for a dynamic tool command.

    Unsupported or variadic parameters are omitted from the direct CLI surface
    and can still be supplied through `--call`.

    Args:
        func (Callable): Tool function.

    Returns:
        inspect.Signature: Signature safe to expose through Typer.
    """
    sig = inspect.signature(func)
    params = []
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect.Parameter.empty:
            param = param.replace(default=typer.Argument(None))
        elif not isinstance(param.default, (typer.models.ArgumentInfo, typer.models.OptionInfo)):
            param = param.replace(default=typer.Option(param.default))
        params.append(param)

    if "call" not in sig.parameters:
        params.append(
            inspect.Parameter(
                "call",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=tp.Optional[str],
                default=typer.Option(None, "--call", help=CALL_HELP),
            )
        )
    return inspect.Signature(parameters=params)


def wrap_tool_command(func: tp.Callable) -> tp.Callable:
    """Wrap a tool function so the CLI prints its return value.

    Args:
        func (Callable): Tool function.

    Returns:
        Callable: Wrapped Typer command.
    """

    @functools.wraps(func)
    def command(*args, **kwargs) -> None:
        ctx = click.get_current_context()
        call = ctx.params.get("call")
        cli_overrides = get_cli_overrides(ctx, exclude={"call"})
        merged_args, merged_kwargs = merge_call_arguments(func, call=call, cli_overrides=cli_overrides)
        validate_required_arguments(func, merged_args, merged_kwargs, ctx)
        print_output(func(*merged_args, **merged_kwargs))

    command.__signature__ = build_tool_signature(func)
    return command


def run_chat_command(func: tp.Callable, call: tp.Optional[str], cli_overrides: tp.Dict[str, tp.Any]) -> None:
    """Run a chat-like command with merged arguments.

    This function intentionally avoids overriding the formatter so the configured
    chat formatter can stream output directly to the terminal in interactive
    fashion.

    Args:
        func (Callable): Target `chat`-like function.
        call (Optional[str]): Structured payload string.
        cli_overrides (Dict[str, Any]): Explicit CLI arguments.

    Returns:
        None
    """
    from vectorbtpro.utils.pbar import ProgressHidden

    ctx = click.get_current_context()
    args, kwargs = merge_call_arguments(func, call=call, cli_overrides=cli_overrides)
    validate_required_arguments(func, args, kwargs, ctx)
    progress = kwargs.pop("progress", False)
    if progress:
        func(*args, **kwargs)
    else:
        with ProgressHidden():
            func(*args, **kwargs)


def build_mcp_app() -> typer.Typer:
    """Build the `mcp` command group.

    Returns:
        typer.Typer: Nested Typer application for MCP commands.
    """
    mcp_app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        pretty_exceptions_enable=False,
        help="Run the MCP server or invoke MCP tools directly.",
    )

    @mcp_app.command("serve")
    def mcp_serve_command(
        transport: tp.Optional[str] = typer.Option(
            None,
            "--transport",
            help="Transport type.",
            case_sensitive=False,
            show_choices=True,
        ),
        call: tp.Optional[str] = typer.Option(None, "--call", help=CALL_HELP),
    ) -> None:
        """Run the MCP server."""
        call_args, call_kwargs = parse_call_payload(call)
        argv = []
        if len(call_args) > 0:
            argv.extend(str(arg) for arg in call_args)
        if "transport" in call_kwargs and transport is None:
            transport = call_kwargs["transport"]
        if transport is not None:
            argv.extend(["--transport", str(transport)])
        mcp_main(argv=argv)

    for tool_name, tool in tool_registry.items():
        command_name = tool_name.replace("_", "-")
        mcp_app.command(name=command_name, help=format_cli_help(inspect.getdoc(tool)))(wrap_tool_command(tool))

    return mcp_app


def build_wrapped_command_help(command_name: str) -> str:
    """Build help text for a wrapped knowledge command.

    Args:
        command_name (str): Name of the function in `vectorbtpro.knowledge.custom_assets`.

    Returns:
        str: CLI help text including the original function docstring.
    """
    from vectorbtpro.knowledge import custom_assets

    func = getattr(custom_assets, command_name)
    doc = inspect.getdoc(func)
    first_line, _, rest = doc.partition("\n")
    parts = []
    if first_line:
        parts.append(first_line)
    parts.append(
        f"CLI wrapper around `vectorbtpro.knowledge.custom_assets.{command_name}`.\n"
        "All well-behaved parameters are exposed directly on the command line, and\n"
        "additional arguments can be supplied via `--call`. The original function docstring:"
    )
    if rest.strip():
        parts.append(rest.strip())
    return "\n\n".join(parts)


@cli_command("chat", help=lambda: build_wrapped_command_help("chat"))
def chat_command(
    ctx: typer.Context,
    query: tp.Optional[str] = typer.Argument(None, help="Chat query."),
    progress: bool = typer.Option(
        False,
        "--progress/--no-progress",
        help="Show progress bars while processing the query.",
    ),
    call: tp.Optional[str] = typer.Option(None, "--call", help=CALL_HELP),
) -> None:
    """Ask VectorBT PRO using asset context via the shell."""
    from vectorbtpro.knowledge.custom_assets import chat

    run_chat_command(chat, call=call, cli_overrides=get_cli_overrides(ctx, exclude={"call"}))


@cli_command("quick-chat", help=lambda: build_wrapped_command_help("quick_chat"))
def quick_chat_command(
    ctx: typer.Context,
    query: tp.Optional[str] = typer.Argument(None, help="Chat query."),
    progress: bool = typer.Option(
        False,
        "--progress/--no-progress",
        help="Show progress bars while processing the query.",
    ),
    call: tp.Optional[str] = typer.Option(None, "--call", help=CALL_HELP),
) -> None:
    """Ask VectorBT PRO using the quick chat preset via the shell."""
    from vectorbtpro.knowledge.custom_assets import quick_chat

    run_chat_command(quick_chat, call=call, cli_overrides=get_cli_overrides(ctx, exclude={"call"}))


@cli_command("interact", help=lambda: build_wrapped_command_help("interact"))
def interact_command(
    ctx: typer.Context,
    query: tp.Optional[str] = typer.Argument(None, help="Chat query."),
    tools: str = typer.Option(
        "all",
        help='Tools to expose to the model, such as "all", "mcp", or "registry".',
    ),
    attach_context: bool = typer.Option(
        False,
        "--attach-context/--no-attach-context",
        help="Attach retrieval context in addition to tools.",
    ),
    progress: bool = typer.Option(
        False,
        "--progress/--no-progress",
        help="Show progress bars while processing the query.",
    ),
    call: tp.Optional[str] = typer.Option(None, "--call", help=CALL_HELP),
) -> None:
    """Ask VectorBT PRO with tool use enabled via the shell."""
    from vectorbtpro.knowledge.custom_assets import interact

    run_chat_command(interact, call=call, cli_overrides=get_cli_overrides(ctx, exclude={"call"}))


def build_app() -> typer.Typer:
    """Build the top-level Typer application.

    Returns:
        typer.Typer: Configured top-level Typer application.
    """
    new_app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        pretty_exceptions_enable=False,
        help="VectorBT PRO command-line interface.",
    )
    for command in cli_registry:
        new_app.command(name=command.name, help=resolve_cli_help(command.help), **command.kwargs)(command.func)
    new_app.add_typer(build_mcp_app(), name="mcp")
    return new_app


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> None:
    """Run the VectorBT PRO CLI.

    Args:
        argv (Optional[Sequence[str]]): Command-line arguments to parse.

    Returns:
        None
    """
    try:
        build_app()(args=list(argv) if argv is not None else None, prog_name="vbt", standalone_mode=False)
    except click.exceptions.NoArgsIsHelpError:
        return
    except click.ClickException as e:
        e.show()
        raise SystemExit(e.exit_code)


if __name__ == "__main__":
    main()
