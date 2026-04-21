# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing MCP tool definitions and the tool registry.

Tools registered here are used by the MCP server (`vectorbtpro.mcp_server`) and the CLI (`vectorbtpro.cli`).
"""

from contextlib import nullcontext

from vectorbtpro import _typing as tp

__all__ = [
    "register_tool",
    "tool_registry",
]


tool_registry = {}
"""Registry mapping tool names to functions for execution."""


def register_tool(arg: tp.Union[None, str, tp.Callable] = None, /, *, name: tp.Optional[str] = None) -> tp.Callable:
    """Decorator to register a function in `tool_registry`.

    Args:
        arg (Union[None, str, Callable]): Tool function or its name.
        name (Optional[str]): Custom name for the tool (if not using the function name).

    Returns:
        Callable: Registered tool function.
    """
    if isinstance(arg, str) and name is None:
        name = arg
        arg = None

    def wrapper(func):
        tool_name = name or func.__name__
        tool_registry[tool_name] = func
        return func

    if callable(arg):
        return wrapper(arg)
    return wrapper


def auto_cast(value: tp.Any) -> tp.Any:
    """Automatically cast a string to an appropriate Python literal type."""
    import ast

    if value is not None and isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


@register_tool
def search(
    query: str,
    asset_names: tp.Optional[tp.List[str]] = None,
    search_method: str = "hybrid",
    with_fallback: bool = True,
    rerank: bool = False,
    rerank_limit: int = 20,
    return_chunks: bool = True,
    return_metadata: str = "minimal",
    dump_engine: tp.Optional[str] = "json",
    max_tokens: tp.Optional[int] = 4_000,
    n: tp.Optional[int] = None,
    page: int = 1,
    progress: bool = False,
) -> str:
    """Search for information relevant to the query.

    !!! note
        This tool is designed to search for general information about VectorBT PRO (vectorbtpro, VBT).
        For specific information about a specific object (such as `vbt.Portfolio`),
        use tools that take a reference name. They operate on the actual objects.

        Also, running this tool on any combination of assets for the first time may take a while,
        as it prepares and caches the documents. If the tool times out repeatedly,
        it's recommended to call `vbt.search("")` directly in your code to build the index
        and then use the MCP tool to search the index.

    Args:
        query (str): Search query.

            Do not reinstate the name "VectorBT PRO" in the query, as it is already implied.
        asset_names (Optional[List[str]]): Asset names to search. Supported names:

            * "api": API reference. Best for specific API queries.
            * "docs": Regular documentation, including getting started, features, tutorials,
                guides, recipes, and legal information. Best for general queries.
            * "messages": Discord messages and discussions. Best for support queries.
            * "examples": Code examples across all assets. Best for practical implementation queries.

            Order doesn't matter.

            Defaults to all.
        search_method (str): Strategy for document search. Supported strategies:

            * "bm25": Uses BM25 for lexical search. Best for specific keywords.
            * "embeddings": Uses embeddings for semantic search. Best for general queries.
            * "hybrid": Combines both embeddings and BM25. Best for balanced search.

            Defaults to "hybrid".
        with_fallback (bool): Whether to fallback to class search if some embeddings are not available;
            otherwise, missing embeddings will be generated, which may take longer.

            Defaults to True.
        rerank (bool): Whether to rerank top results using a cross-encoder for better relevance.

            May require installation, additional resources, and time. Defaults to False.
        rerank_limit (int): Number of top results to rerank.

            Others will be removed. Defaults to 20.
        return_chunks (bool): Whether to return the chunks of the results; otherwise, returns the full results.

            Defaults to True.
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as title and URL.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "minimal".
        dump_engine (Optional[str]): Engine used to serialize results to strings.

            * Use "json" for fast, LLM-friendly output.
            * Use "text" for human-friendly output.
            * Use None for the global default.
            * Also supports "yaml" and other engines.

            Defaults to "json".
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.
        n (Optional[int]): Number of results to return per page.

            If specified, it will return the first `n` results that fit within the `max_tokens` limit.
            If None, returns all results that fit within the `max_tokens` limit. Defaults to None.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.
        progress (bool): Show progress bars while processing the query.

    Returns:
        str: Context string containing the search results.
    """
    from vectorbtpro.knowledge.custom_assets import search
    from vectorbtpro.utils.pbar import ProgressHidden

    with nullcontext() if progress else ProgressHidden():
        query = auto_cast(query)
        asset_names = auto_cast(asset_names)
        search_method = auto_cast(search_method)
        return_chunks = auto_cast(return_chunks)
        return_metadata = auto_cast(return_metadata)
        dump_engine = auto_cast(dump_engine)
        max_tokens = auto_cast(max_tokens)
        n = auto_cast(n)
        page = auto_cast(page)

        if asset_names is None:
            asset_names = "all"
        if with_fallback and search_method == "embeddings":
            search_method = "embeddings_fallback"
        elif with_fallback and search_method == "hybrid":
            search_method = "hybrid_fallback"

        results = search(
            query,
            search_method=search_method,
            return_chunks=return_chunks,
            find_assets_kwargs=dict(
                asset_names=asset_names,
                minimize=False,
            ),
            rerank=rerank,
            rerank_limit=rerank_limit,
            display=False,
        )
        return results.to_context(
            return_metadata=return_metadata,
            dump_engine=dump_engine,
            max_tokens=max_tokens,
            n=n,
            page=page,
        )


@register_tool
def resolve_refnames(
    refnames: tp.List[str],
    dump_engine: tp.Optional[str] = "json",
    max_tokens: tp.Optional[int] = 4_000,
    n: tp.Optional[int] = None,
    page: int = 1,
) -> str:
    """Resolve reference names to their fully qualified names.

    Output format:

    * Success: `OK <input> <resolved>`
    * Failure: `FAIL <input>`

    Args:
        refnames (List[str]): Reference names to resolve.

            A reference name may be a fully qualified dotted path ("vectorbtpro.data.base.Data"),
            a library re-export ("vectorbtpro.Data"), a common alias ("vbt.Data"),
            or a simple name ("Data") that uniquely identifies an object.
        dump_engine (Optional[str]): Engine used to serialize results to strings.

            * Use "json" for fast, LLM-friendly output.
            * Use "text" for human-friendly output.
            * Use None for the global default.
            * Also supports "yaml" and other engines.

            Defaults to "json".
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.
        n (Optional[int]): Number of results to return per page.

            If specified, it will return the first `n` results that fit within the `max_tokens` limit.
            If None, returns all results that fit within the `max_tokens` limit. Defaults to None.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.

    Returns:
        str: Output string containing the resolution results.
    """
    from vectorbtpro.knowledge.custom_assets import VBTAsset
    from vectorbtpro.utils.refs import resolve_refname

    dump_engine = auto_cast(dump_engine)
    max_tokens = auto_cast(max_tokens)
    n = auto_cast(n)
    page = auto_cast(page)

    output = []
    for refname in refnames:
        refname = auto_cast(refname)
        resolved_refname = resolve_refname(refname)
        if resolved_refname:
            output.append(f"OK {refname} {resolved_refname}")
        else:
            output.append(f"FAIL {refname}")
    return VBTAsset(output).to_context(
        return_metadata="none",
        dump_engine=dump_engine,
        max_tokens=max_tokens,
        n=n,
        page=page,
    )


@register_tool
def find(
    refnames: tp.List[str],
    resolve: bool = True,
    asset_names: tp.Optional[tp.List[str]] = None,
    aggregate_api: bool = False,
    aggregate_messages: bool = False,
    return_metadata: str = "minimal",
    dump_engine: tp.Optional[str] = "json",
    max_tokens: tp.Optional[int] = 4_000,
    n: tp.Optional[int] = None,
    page: int = 1,
    progress: bool = False,
) -> str:
    """Find information relevant to specific objects.

    This can be used to find assets mentioning specific VectorBT PRO (vectorbtpro, VBT) objects,
    such as modules, classes, functions, and instances. For example, searching for "Portfolio"
    will generate targets such as `vbt.Portfolio`, `Portfolio(...)`, `pf = ...`, etc.

    If any of the mentioned targets are found in an asset, it will be returned.

    !!! note
        All references must be valid; if any reference cannot be resolved, will raise an error.
        Thus, when passing multiple references, use `resolve_refnames` to verify them first.

    Args:
        refnames (List[str]): Reference names of the objects.

            A reference name may be a fully qualified dotted path ("vectorbtpro.data.base.Data"),
            a library re-export ("vectorbtpro.Data"), a common alias ("vbt.Data"),
            or a simple name ("Data") that uniquely identifies an object.

            Returns a code example if any of the references are found in the code example.
        resolve (bool): Whether to resolve the reference to an actual object.

            Set to False to find any string, not just VBT objects, such as "SQLAlchemy".
            In this case, `refname` becomes a simple string to match against.
            Defaults to True.
        asset_names (Optional[List[str]]): Asset names to search. Supported names:

            * "api": API reference.
            * "docs": Regular documentation, including getting started, features, tutorials,
                guides, recipes, and legal information.
            * "messages": Discord messages and discussions.
            * "examples": Code examples across all assets.
            * "all": All of the above, in the order specified above.

            Order matters. May also include ellipsis. For example, `["messages", "..."]` puts
            "messages" at the beginning and all other assets in their usual order at the end.

            Defaults to all.
        aggregate_api (bool): Whether to aggregate all children of the object into a single context.

            If True, the context will contain all the children of the object, such as methods,
            properties, and attributes, in a single context string. Note that this might result in
            a large context string, especially for modules and classes. If False, the context
            will contain only the object description.

            Applies only to API documentation. Defaults to False.
        aggregate_messages (bool): Whether to aggregate messages belonging to the same thread (question-reply chain).

            If True, finding an object in a message will return the question and all replies
            in a single context string, not just the isolated message containing the object.

            Applies only to Discord messages. Defaults to False.
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as title and URL.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "minimal".
        dump_engine (Optional[str]): Engine used to serialize results to strings.

            * Use "json" for fast, LLM-friendly output.
            * Use "text" for human-friendly output.
            * Use None for the global default.
            * Also supports "yaml" and other engines.

            Defaults to "json".
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.
        n (Optional[int]): Number of results to return per page.

            If specified, it will return the first `n` results that fit within the `max_tokens` limit.
            If None, returns all results that fit within the `max_tokens` limit. Defaults to None.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.
        progress (bool): Show progress bars while processing the query.

    Returns:
        str: Context string containing the search results.
    """
    from vectorbtpro.knowledge.custom_assets import find_assets
    from vectorbtpro.utils.pbar import ProgressHidden

    with nullcontext() if progress else ProgressHidden():
        refnames = auto_cast(refnames)
        resolve = auto_cast(resolve)
        asset_names = auto_cast(asset_names)
        aggregate_api = auto_cast(aggregate_api)
        aggregate_messages = auto_cast(aggregate_messages)
        return_metadata = auto_cast(return_metadata)
        dump_engine = auto_cast(dump_engine)
        max_tokens = auto_cast(max_tokens)
        n = auto_cast(n)
        page = auto_cast(page)

        if asset_names is None:
            asset_names = "all"

        results = find_assets(
            refnames,
            resolve=resolve,
            asset_names=asset_names,
            api_kwargs=dict(
                only_obj=True,
                aggregate=aggregate_api,
            ),
            docs_kwargs=dict(
                aggregate=False,
                up_aggregate=False,
            ),
            messages_kwargs=dict(
                aggregate="threads" if aggregate_messages else "messages",
                latest_first=True,
            ),
            examples_kwargs=dict(
                return_type="field" if return_metadata.lower() == "none" else "item",
                latest_first=True,
            ),
            minimize=False,
        )
        return results.to_context(
            return_metadata=return_metadata,
            dump_engine=dump_engine,
            max_tokens=max_tokens,
            n=n,
            page=page,
        )


@register_tool
def get_page(
    url: str,
    return_metadata: str = "none",
    dump_engine: tp.Optional[str] = "json",
    max_tokens: tp.Optional[int] = 4_000,
) -> str:
    """Get the content of a documentation page by its URL.

    Args:
        url (str): URL of the documentation page. Supported formats:

            * Full URL (e.g., "https://vectorbt.pro/.../getting-started/installation/").
            * Relative path (e.g., "/getting-started/installation/").
            * Fragment identifier (e.g., "getting-started/installation/#windows").
            * Fully-qualified dotted name of a VBT object (e.g., "vectorbtpro.data.base.Data").
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as title and URL.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "none".
        dump_engine (Optional[str]): Engine used to serialize results to strings.

            * Use "json" for fast, LLM-friendly output.
            * Use "text" for human-friendly output.
            * Use None for the global default.
            * Also supports "yaml" and other engines.

            Defaults to "json".
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.

    Returns:
        str: Content of the documentation page.
    """
    from vectorbtpro.knowledge.custom_assets import PagesAsset

    return_metadata = auto_cast(return_metadata)
    dump_engine = auto_cast(dump_engine)
    max_tokens = auto_cast(max_tokens)

    results = PagesAsset.pull().find_page(url, aggregate=True)
    return results.to_context(
        return_metadata=return_metadata,
        dump_engine=dump_engine,
        max_tokens=max_tokens,
    )


@register_tool
def get_message(
    url: str,
    return_metadata: str = "none",
    dump_engine: tp.Optional[str] = "json",
    max_tokens: tp.Optional[int] = 4_000,
) -> str:
    """Get the content of a Discord message by its URL.

    Args:
        url (str): URL of the Discord message.
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as channel and author.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "none".
        dump_engine (Optional[str]): Engine used to serialize results to strings.

            * Use "json" for fast, LLM-friendly output.
            * Use "text" for human-friendly output.
            * Use None for the global default.
            * Also supports "yaml" and other engines.

            Defaults to "json".
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.

    Returns:
        str: Content of the Discord message.
    """
    from vectorbtpro.knowledge.custom_assets import MessagesAsset

    return_metadata = auto_cast(return_metadata)
    dump_engine = auto_cast(dump_engine)
    max_tokens = auto_cast(max_tokens)

    results = MessagesAsset.pull().find_link(url)
    return results.to_context(
        return_metadata=return_metadata,
        dump_engine=dump_engine,
        max_tokens=max_tokens,
    )


@register_tool
def get_message_block(
    url: str,
    return_metadata: str = "none",
    dump_engine: tp.Optional[str] = "json",
    max_tokens: tp.Optional[int] = 4_000,
    n: tp.Optional[int] = None,
    page: int = 1,
) -> str:
    """Get the content of a Discord message block by its URL.

    A block is a group of messages sent by the same user in a short time frame.
    The URL of a block is the same as the URL of the first message in the block.

    Args:
        url (str): URL of the Discord message block.
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as channel and author.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "none".
        dump_engine (Optional[str]): Engine used to serialize results to strings.

            * Use "json" for fast, LLM-friendly output.
            * Use "text" for human-friendly output.
            * Use None for the global default.
            * Also supports "yaml" and other engines.

            Defaults to "json".
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.
        n (Optional[int]): Number of results to return per page.

            If specified, it will return the first `n` results that fit within the `max_tokens` limit.
            If None, returns all results that fit within the `max_tokens` limit. Defaults to None.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.

    Returns:
        str: Content of the Discord message block.
    """
    from vectorbtpro.knowledge.custom_assets import MessagesAsset

    return_metadata = auto_cast(return_metadata)
    dump_engine = auto_cast(dump_engine)
    max_tokens = auto_cast(max_tokens)
    n = auto_cast(n)
    page = auto_cast(page)

    results = MessagesAsset.pull().find_link(url, field="block", single_item=False)
    return results.to_context(
        return_metadata=return_metadata,
        dump_engine=dump_engine,
        max_tokens=max_tokens,
        n=n,
        page=page,
    )


@register_tool
def get_message_thread(
    url: str,
    return_metadata: str = "none",
    dump_engine: tp.Optional[str] = "json",
    max_tokens: tp.Optional[int] = 4_000,
    n: tp.Optional[int] = None,
    page: int = 1,
) -> str:
    """Get the content of a Discord message thread by its URL.

    A thread is a question-reply chain. The URL of a thread is the same as the URL of the
    initial message in the thread.

    Args:
        url (str): URL of the Discord message thread.
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as channel and author.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "none".
        dump_engine (Optional[str]): Engine used to serialize results to strings.

            * Use "json" for fast, LLM-friendly output.
            * Use "text" for human-friendly output.
            * Use None for the global default.
            * Also supports "yaml" and other engines.

            Defaults to "json".
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.
        n (Optional[int]): Number of results to return per page.

            If specified, it will return the first `n` results that fit within the `max_tokens` limit.
            If None, returns all results that fit within the `max_tokens` limit. Defaults to None.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.

    Returns:
        str: Content of the Discord message thread.
    """
    from vectorbtpro.knowledge.custom_assets import MessagesAsset

    return_metadata = auto_cast(return_metadata)
    dump_engine = auto_cast(dump_engine)
    max_tokens = auto_cast(max_tokens)
    n = auto_cast(n)
    page = auto_cast(page)

    results = MessagesAsset.pull().find_link(url, field="thread", single_item=False)
    return results.to_context(
        return_metadata=return_metadata,
        dump_engine=dump_engine,
        max_tokens=max_tokens,
        n=n,
        page=page,
    )


@register_tool
def get_attrs(
    refname: str,
    own_only: bool = False,
    incl_private: bool = False,
    incl_types: bool = False,
    incl_refnames: bool = False,
    dump_engine: tp.Optional[str] = "json",
    max_tokens: tp.Optional[int] = 4_000,
    n: tp.Optional[int] = None,
    page: int = 1,
) -> str:
    """Get a list of attributes of an object with their types and reference names.

    Similar to `dir()`, but with more information and better formatting.
    Can be used to discover the API of VectorBT PRO (vectorbtpro, VBT). For example, use it to
    find out what methods and properties are available on a specific class, or to explore the
    objects defined in a module.

    Each line is formatted as `<name> [<type>] (@ <refname>)`, where the `@ <refname>` suffix
    is shown only when the attribute is not defined directly on the object.

    Args:
        refname (str): Reference name of the object.

            A reference name may be a fully qualified dotted path ("vectorbtpro.data.base.Data"),
            a library re-export ("vectorbtpro.Data"), a common alias ("vbt.Data"),
            or a simple name ("Data") that uniquely identifies an object.

            Pass "vbt" to get all the attributes of the `vectorbtpro` module.
        own_only (bool): If True, include only attributes that are defined directly on the object
            (i.e., attributes defined elsewhere, such as inherited attributes, will be excluded).
        incl_private (bool): If True, include private attributes (those starting with an underscore).
        incl_types (bool): If True, include attribute types in the output (e.g., `classmethod`).
        incl_refnames (bool): If True, include attribute reference names in the output
            (e.g., `vectorbtpro.utils.base.Base.chat`).
        dump_engine (Optional[str]): Engine used to serialize results to strings.

            * Use "json" for fast, LLM-friendly output.
            * Use "text" for human-friendly output.
            * Use None for the global default.
            * Also supports "yaml" and other engines.

            Defaults to "json".
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.
        n (Optional[int]): Number of results to return per page.

            If specified, it will return the first `n` results that fit within the `max_tokens` limit.
            If None, returns all results that fit within the `max_tokens` limit. Defaults to None.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.

    Returns:
        str: String containing the list of attributes, each on a new line.
    """
    from vectorbtpro.knowledge.custom_assets import VBTAsset
    from vectorbtpro.utils.attr_ import get_attrs
    from vectorbtpro.utils.refs import resolve_refname, get_refname_obj

    refname = auto_cast(refname)
    dump_engine = auto_cast(dump_engine)
    max_tokens = auto_cast(max_tokens)
    n = auto_cast(n)
    page = auto_cast(page)

    resolved_refname = resolve_refname(refname)
    if not resolved_refname:
        raise ValueError(f"Reference name {refname!r} cannot be resolved to an object")
    obj = get_refname_obj(resolved_refname)
    attr_meta = get_attrs(obj=obj, own_only=own_only, incl_private=incl_private, return_meta=True)

    display_lines = []
    for m in attr_meta:
        line = m.name
        if incl_types and m.type != "?":
            line += f" [{m.type}]"
        if incl_refnames and m.refname != "?" and m.refname != resolved_refname + "." + m.name:
            line += f" @ {m.refname}"
        display_lines.append(line)
    return VBTAsset(display_lines).to_context(
        return_metadata="none",
        dump_engine=dump_engine,
        max_tokens=max_tokens,
        n=n,
        page=page,
    )


@register_tool
def get_source(
    refname: str,
    max_tokens: tp.Optional[int] = 4_000,
) -> str:
    """Get the source code of any object.

    This can be used to inspect the implementation of VectorBT PRO (vectorbtpro, VBT) objects,
    such as modules, classes, functions, and instances. It uses AST parsing to retrieve the source code
    of any object, including named tuples, class variables, dataclasses, and other objects that
    may not have a traditional source code representation.

    Args:
        refname (str): Reference name of the object.

            A reference name may be a fully qualified dotted path ("vectorbtpro.data.base.Data"),
            a library re-export ("vectorbtpro.Data"), a common alias ("vbt.Data"),
            or a simple name ("Data") that uniquely identifies an object.
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.

    Returns:
        str: Source code of the object.

            Multiple references can be provided, in which case the source code of each object
            is concatenated together, separated by two newlines.
    """
    from vectorbtpro.knowledge.custom_assets import VBTAsset
    from vectorbtpro.utils.source import get_source
    from vectorbtpro.utils.refs import resolve_refname

    refname = auto_cast(refname)
    max_tokens = auto_cast(max_tokens)

    resolved_refname = resolve_refname(refname)
    if not resolved_refname:
        raise ValueError(f"Reference name {refname!r} cannot be resolved to an object")
    return VBTAsset([get_source(resolved_refname)]).to_context(
        return_metadata="none",
        max_tokens=max_tokens,
    )


current_kernel = None
"""Currently running Jupyter kernel for executing code snippets."""


@register_tool
def run_code(
    code: str,
    restart: bool = False,
    exec_timeout: tp.Optional[float] = None,
    max_tokens: tp.Optional[int] = 4_000,
) -> str:
    """Run a code snippet with all the necessary imports and return the output.

    This spins up a Jupyter kernel if it is not already running, and automatically imports
    `from vectorbtpro import *`, which includes `vbt`, `pd` (Pandas), `np` (NumPy), `njit` (from Numba),
    and other commonly used modules from the documentation. Running this the second time will
    reuse the existing kernel and all variables defined earlier.

    Use this tool to develop and test code snippets in the VectorBT PRO (vectorbtpro, VBT) environment,
    similar to a Jupyter notebook. You can backtest a strategy, debug a function, or explore data interactively.

    !!! note
        VBT is centered around easy-to-use APIs and high-performance computing, thus you should
        put priority on discovering and using VBT APIs. Before running this tool, use other MCP tools to
        search for relevant information, such as existing code examples, API references, and documentation.
        This will help you understand how to use VBT effectively and avoid reinventing the wheel.
        If a custom implementation is needed, consider extending existing VBT functionality or
        using high-performance libraries such as Numba.

    !!! warning
        Ensure that the code is safe, as this tool can execute arbitrary code in the current environment.
        Do not run code that has side effects, such as installing new dependencies, modifying global state,
        or performing I/O operations, unless explicitly granted permission!

        Be aware of long-running and resource-intensive code snippets, as they may block the kernel
        and prevent further execution. Use minimal code snippets whenever possible.

        Use this tool for development and testing purposes related to VBT only.

    Args:
        code (str): Code snippet to run.
        restart (bool): Whether to restart the kernel before running the code.

            Defaults to False.
        exec_timeout (Optional[float]): Timeout for the code execution in seconds.

            None means no timeout. Defaults to None.
        max_tokens (Optional[int]): Maximum number of tokens to return.

            Defaults to 4,000.

    Returns:
        str: Output of the executed code.
    """
    from vectorbtpro.knowledge.custom_assets import VBTAsset
    from vectorbtpro.utils.eval_ import VBTKernel

    code = auto_cast(code)
    restart = auto_cast(restart)
    exec_timeout = auto_cast(exec_timeout)
    max_tokens = auto_cast(max_tokens)

    global current_kernel
    if current_kernel is None:
        current_kernel = VBTKernel()
        current_kernel.start()
    if restart:
        current_kernel.restart()
    output = current_kernel.execute(code, exec_timeout=exec_timeout, raise_on_error=True)
    return VBTAsset([output]).to_context(
        return_metadata="none",
        max_tokens=max_tokens,
    )
