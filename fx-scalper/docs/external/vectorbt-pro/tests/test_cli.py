import typing

from typer.testing import CliRunner

from vectorbtpro import cli


def test_chat_command(monkeypatch, capsys):
    called = {}

    def fake_run_chat_command(func, call=None, cli_overrides=None):
        called["func"] = func.__name__
        called["call"] = call
        called["cli_overrides"] = cli_overrides
        print("chat-output")

    monkeypatch.setattr(cli, "run_chat_command", fake_run_chat_command)

    cli.main(["chat", "What is PFO?"])
    captured = capsys.readouterr()

    assert captured.out == "chat-output\n"
    assert called["func"] == "chat"
    assert called["call"] is None
    assert called["cli_overrides"] == {"query": "What is PFO?"}


def test_interact_command(monkeypatch, capsys):
    called = {}

    def fake_run_chat_command(func, call=None, cli_overrides=None):
        called["func"] = func.__name__
        called["call"] = call
        called["cli_overrides"] = cli_overrides
        print("interact-output")

    monkeypatch.setattr(cli, "run_chat_command", fake_run_chat_command)

    cli.main(["interact", "Use tools", "--tools", "mcp", "--attach-context", "--progress"])
    captured = capsys.readouterr()

    assert captured.out == "interact-output\n"
    assert called["func"] == "interact"
    assert called["call"] is None
    assert called["cli_overrides"] == {"query": "Use tools", "tools": "mcp", "attach_context": True, "progress": True}


def test_quick_chat_command(monkeypatch, capsys):
    called = {}

    def fake_run_chat_command(func, call=None, cli_overrides=None):
        called["func"] = func.__name__
        called["call"] = call
        called["cli_overrides"] = cli_overrides
        print("quick-chat-output")

    monkeypatch.setattr(cli, "run_chat_command", fake_run_chat_command)

    cli.main(["quick-chat", "What is PFO?"])
    captured = capsys.readouterr()

    assert captured.out == "quick-chat-output\n"
    assert called["func"] == "quick_chat"
    assert called["call"] is None
    assert called["cli_overrides"] == {"query": "What is PFO?"}


def test_registered_tool_command(monkeypatch, capsys):
    called = {}

    def fake_tool(
        query: str,
        asset_names: typing.Optional[typing.List[str]] = None,
        rerank: bool = False,
        return_chunks: bool = True,
    ) -> str:
        called["query"] = query
        called["asset_names"] = asset_names
        called["rerank"] = rerank
        called["return_chunks"] = return_chunks
        return "tool-output"

    monkeypatch.setitem(cli.tool_registry, "fake_tool", fake_tool)

    cli.main(
        [
            "mcp",
            "fake-tool",
            "PFO",
            "--asset-names",
            "docs",
            "--asset-names",
            "api",
            "--rerank",
            "--no-return-chunks",
        ]
    )
    captured = capsys.readouterr()

    assert captured.out == "tool-output\n"
    assert called == {
        "query": "PFO",
        "asset_names": ["docs", "api"],
        "rerank": True,
        "return_chunks": False,
    }


def test_registered_tool_hyphen_name(monkeypatch, capsys):
    called = {}

    def fake_tool(refnames: typing.List[str]) -> str:
        called["refnames"] = refnames
        return "alias-output"

    monkeypatch.setitem(cli.tool_registry, "fake_alias_tool", fake_tool)

    cli.main(["mcp", "fake-alias-tool", "Portfolio", "Data"])
    captured = capsys.readouterr()

    assert captured.out == "alias-output\n"
    assert called["refnames"] == ["Portfolio", "Data"]


def test_registered_tool_call_payload(monkeypatch, capsys):
    called = {}

    def fake_tool(query: str, rerank: bool = False, **kwargs) -> str:
        called["query"] = query
        called["rerank"] = rerank
        called["kwargs"] = kwargs
        return "call-output"

    monkeypatch.setitem(cli.tool_registry, "fake_tool_call", fake_tool)

    cli.main(
        [
            "mcp",
            "fake-tool-call",
            "--call",
            '{"args": ["PFO"], "kwargs": {"rerank": true, "limit": 5}}',
        ]
    )
    captured = capsys.readouterr()

    assert captured.out == "call-output\n"
    assert called == {"query": "PFO", "rerank": True, "kwargs": {"limit": 5}}


def test_registered_tool_call_kwargs(monkeypatch, capsys):
    called = {}

    def fake_tool(query: str, rerank: bool = False) -> str:
        called["query"] = query
        called["rerank"] = rerank
        return "kwargs-output"

    monkeypatch.setitem(cli.tool_registry, "fake_tool_kwargs_call", fake_tool)

    cli.main(
        [
            "mcp",
            "fake-tool-kwargs-call",
            "--call",
            '{"query": "hello!", "rerank": true}',
        ]
    )
    captured = capsys.readouterr()

    assert captured.out == "kwargs-output\n"
    assert called == {"query": "hello!", "rerank": True}


def test_registered_tool_missing_arg(monkeypatch):
    runner = CliRunner()

    def fake_tool(query: str, rerank: bool = False) -> str:
        return f"{query}:{rerank}"

    monkeypatch.setitem(cli.tool_registry, "fake_required_tool", fake_tool)

    result = runner.invoke(cli.build_app(), ["mcp", "fake-required-tool"])

    assert result.exit_code != 0
    assert "Usage:" in result.output
    assert "Missing argument" in result.output


def test_main_formats_usage_error(monkeypatch, capsys):
    def fake_tool(query: str, rerank: bool = False) -> str:
        return f"{query}:{rerank}"

    monkeypatch.setitem(cli.tool_registry, "fake_required_tool_main", fake_tool)

    try:
        try:
            cli.main(["mcp", "fake-required-tool-main"])
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("Expected SystemExit")
    finally:
        del cli.tool_registry["fake_required_tool_main"]

    captured = capsys.readouterr()
    assert "Usage: vbt mcp fake-required-tool-main" in captured.err
    assert "Error: Missing argument" in captured.err


def test_mcp_command_forwards_transport(monkeypatch):
    called = {}

    def fake_mcp_main(argv=None):
        called["argv"] = argv

    monkeypatch.setattr(cli, "mcp_main", fake_mcp_main)

    cli.main(["mcp", "serve", "--transport", "streamable-http"])

    assert called["argv"] == ["--transport", "streamable-http"]
