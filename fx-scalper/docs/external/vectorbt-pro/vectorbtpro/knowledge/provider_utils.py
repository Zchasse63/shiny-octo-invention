# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for knowledge providers."""

from vectorbtpro import _typing as tp


def check_ollama_available() -> bool:
    """Check if Ollama is installed and its daemon is reachable.

    Returns:
        bool: True if Ollama is available, False otherwise.
    """
    from vectorbtpro.utils.module_ import check_installed

    if not check_installed("ollama"):
        return False
    try:
        from ollama import Client

        Client().list()
        return True
    except Exception:
        return False


def resolve_provider(
    provider: str,
    remote_candidates: tp.Sequence[tp.Tuple[str, tp.Union[bool, tp.Callable]]] = (),
    local_candidates: tp.Sequence[tp.Tuple[str, tp.Union[bool, tp.Callable]]] = (),
    remote_fallback_candidates: tp.Sequence[tp.Tuple[str, tp.Union[bool, tp.Callable]]] = (),
    local_fallback_candidates: tp.Sequence[tp.Tuple[str, tp.Union[bool, tp.Callable]]] = (),
    fallback_candidates: tp.Sequence[tp.Tuple[str, tp.Union[bool, tp.Callable]]] = (),
    provider_name: str = "provider",
) -> tp.Optional[str]:
    """Resolve the provider based on the given mode and candidates.

    Args:
        provider (str): Provider or mode to resolve the provider.

            Supported modes are:

            * "auto": Choose R, then L, then RF, then LF, then F.
            * "prefer_remote": Choose R, then RF, then L, then LF, then F.
            * "prefer_local": Choose L, then LF, then R, then RF, then F.
            * "only_remote": Choose R, then RF.
            * "only_local": Choose L, then LF.
            * Any other value is treated as a provider name.
        remote_candidates (Sequence[Tuple[str, Union[bool, Callable]]]):
            Sequence of remote provider candidates.

            Each candidate is a tuple of (name, is_available), where is_available can be a boolean
            or a callable that returns a boolean.
        local_candidates (Sequence[Tuple[str, Union[bool, Callable]]]):
            Sequence of local provider candidates.

            Each candidate is a tuple of (name, is_available), where is_available can be a boolean
            or a callable that returns a boolean.
        remote_fallback_candidates (Sequence[Tuple[str, Union[bool, Callable]]]):
            Sequence of remote fallback provider candidates.

            Each candidate is a tuple of (name, is_available), where is_available can be a boolean
            or a callable that returns a boolean.
        local_fallback_candidates (Sequence[Tuple[str, Union[bool, Callable]]]):
            Sequence of local fallback provider candidates.

            Each candidate is a tuple of (name, is_available), where is_available can be a boolean
            or a callable that returns a boolean.
        fallback_candidates (Sequence[Tuple[str, Union[bool, Callable]]]):
            Sequence of fallback provider candidates.

            Each candidate is a tuple of (name, is_available), where is_available can be a boolean
            or a callable that returns a boolean.
        provider_name (str): Name of the provider argument for error messages.

    Returns:
        Optional[str]: The name of the resolved provider, or None if no provider is available.
    """
    all_remote = tuple(remote_candidates) + tuple(remote_fallback_candidates)
    all_local = tuple(local_candidates) + tuple(local_fallback_candidates)
    all_candidates = all_remote + all_local + fallback_candidates
    if provider.lower() in (name.lower() for name, _ in all_candidates):
        return provider
    if provider.lower() == "auto":
        candidates = (
            tuple(remote_candidates)
            + tuple(local_candidates)
            + tuple(remote_fallback_candidates)
            + tuple(local_fallback_candidates)
            + tuple(fallback_candidates)
        )
    elif provider.lower() == "prefer_remote":
        candidates = all_remote + all_local + tuple(fallback_candidates)
    elif provider.lower() == "prefer_local":
        candidates = all_local + all_remote + tuple(fallback_candidates)
    elif provider.lower() == "only_remote":
        candidates = all_remote
    elif provider.lower() == "only_local":
        candidates = all_local
    else:
        raise ValueError(f"Invalid {provider_name} or mode: {provider!r}")
    for name, is_available in candidates:
        if callable(is_available):
            is_available = is_available()
        if is_available:
            return name
    return None
