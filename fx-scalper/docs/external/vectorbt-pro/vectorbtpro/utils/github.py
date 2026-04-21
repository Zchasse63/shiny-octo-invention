# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing GitHub-related utilities."""

import io
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import flat_merge_dicts
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.refs import get_caller_qualname

__all__ = []


def resolve_release_name(
    repo_owner: str,
    repo_name: str,
    release_name: tp.Optional[str] = None,
    raise_missing: bool = True,
    token: tp.Optional[str] = None,
    token_required: bool = False,
    use_pygithub: tp.Optional[bool] = None,
) -> tp.Optional[str]:
    """Resolve a release name to a concrete GitHub release tag.

    Args:
        repo_owner (str): Owner of the GitHub repository.
        repo_name (str): Name of the GitHub repository.
        release_name (Optional[str]): Release specification.

            * None or 'current' resolves to the current package release tag.
            * 'latest' resolves to the most recent GitHub release tag.
            * Any other value is returned as-is.
        raise_missing (bool): Whether to raise if no matching release is found.
        token (Optional[str]): GitHub authentication token.
        token_required (bool): Flag indicating whether a GitHub token is required.
        use_pygithub (Optional[bool]): Use the PyGithub library to fetch release data.

            If True, uses https://github.com/PyGithub/PyGithub (otherwise requests).

    Returns:
        Optional[str]: Resolved release tag name, or None if no matching release is found
            and `raise_missing` is False.
    """
    if release_name is None or release_name.lower() == "current":
        from vectorbtpro._version import __release__

        return __release__
    if release_name.lower() == "latest":
        import os

        import requests

        if token is None:
            token = os.environ.get("GITHUB_TOKEN", None)
        if token is None and token_required:
            raise ValueError("GitHub token is required")
        if use_pygithub is None:
            from vectorbtpro.utils.module_ import check_installed

            use_pygithub = check_installed("github")
        if use_pygithub:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("github")
            from github import Github, Auth
            from github.GithubException import UnknownObjectException

            if token is not None:
                g = Github(auth=Auth.Token(token))
            else:
                g = Github()
            try:
                repo = g.get_repo(f"{repo_owner}/{repo_name}")
            except (requests.RequestException, UnknownObjectException):
                if not raise_missing:
                    return None
                raise Exception(f"Repository '{repo_owner}/{repo_name}' not found or access denied")
            try:
                release = repo.get_latest_release()
            except (requests.RequestException, UnknownObjectException):
                if not raise_missing:
                    return None
                raise Exception("Latest release not found")
            return release.title
        else:
            headers = {"Accept": "application/vnd.github+json"}
            if token is not None:
                headers["Authorization"] = f"token {token}"
            release_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
            try:
                response = requests.get(release_url, headers=headers)
                response.raise_for_status()
            except requests.RequestException:
                if not raise_missing:
                    return None
                raise
            release_info = response.json()
            resolved_release_name = release_info.get("name")
            if resolved_release_name is None and not raise_missing:
                return None
            return resolved_release_name
    return release_name


def get_asset_value(asset: tp.Any, key: str) -> tp.Any:
    """Get a value from an asset represented either as an object or a mapping.

    Args:
        asset (Any): Asset represented either as a mapping or as an object exposing a `name` attribute.
        key (str): Key or attribute name to retrieve.

    Returns:
        Any: Value corresponding to the specified key or attribute.
    """
    if isinstance(asset, dict):
        return asset[key]
    return getattr(asset, key)


def resolve_asset_name_from_names(
    asset_names: tp.Iterable[str],
    asset_name: tp.Optional[str] = None,
    release_name: tp.Optional[str] = None,
    raise_missing: bool = True,
) -> tp.Optional[str]:
    """Resolve an asset name from an iterable of candidate asset names.

    Args:
        asset_names (Iterable[str]): Candidate asset names.
        asset_name (Optional[str]): Name of the asset to resolve.

            Matches assets by exact name first. If no exact match is found,
            searches for names extending the requested name with additional
            dot-separated suffixes, such as 'messages' resolving to
            'messages.json' or 'vectorbt.pro' resolving to
            'vectorbt.pro.gz'.
        release_name (Optional[str]): Name of the release used in error messages.
        raise_missing (bool): Whether to raise if no matching asset is found.

    Returns:
        Optional[str]: Resolved asset name, or None if not found and `raise_missing` is False.
    """
    asset_names = list(asset_names)
    if asset_name is None:
        if len(asset_names) == 1:
            return asset_names[0]
        raise Exception("Please specify asset_name")

    if asset_name in asset_names:
        return asset_name

    matching_names = [name for name in asset_names if name.startswith(f"{asset_name}.")]
    if len(matching_names) == 1:
        return matching_names[0]
    if len(matching_names) > 1:
        matching_names_str = ", ".join(repr(name) for name in matching_names)
        if release_name is not None:
            raise Exception(
                f"Multiple assets matching {asset_name!r} found in release {release_name!r}: {matching_names_str}"
            )
        raise Exception(f"Multiple assets matching {asset_name!r} found: {matching_names_str}")
    if raise_missing:
        if release_name is not None:
            raise Exception(f"Asset {asset_name!r} not found in release {release_name!r}")
        raise Exception(f"Asset {asset_name!r} not found")
    return None


def resolve_asset_name(
    repo_owner: str,
    repo_name: str,
    release_name: str,
    asset_name: tp.Optional[str] = None,
    raise_missing: bool = True,
    token: tp.Optional[str] = None,
    token_required: bool = False,
    use_pygithub: tp.Optional[bool] = None,
) -> tp.Optional[str]:
    """Resolve an asset name to a concrete GitHub release asset name.

    Args:
        repo_owner (str): Owner of the GitHub repository.
        repo_name (str): Name of the GitHub repository.
        release_name (str): GitHub release tag (e.g. 'v2024.12.15').

            Must be a concrete release name. Use `resolve_release_name`.
        asset_name (Optional[str]): Name of the asset to resolve.

            Matches assets by exact name first. If no exact match is found,
            searches for assets whose names extend the requested name with
            additional dot-separated suffixes, such as 'messages' resolving
            to 'messages.json' or 'vectorbt.pro' resolving to 'vectorbt.pro.gz'.
        raise_missing (bool): Whether to raise if no matching asset is found.
        token (Optional[str]): GitHub authentication token.
        token_required (bool): Flag indicating whether a GitHub token is required.
        use_pygithub (Optional[bool]): Use the PyGithub library to fetch release data.

            If True, uses https://github.com/PyGithub/PyGithub (otherwise requests).

    Returns:
        Optional[str]: Resolved asset name, or None if not found and `raise_missing` is False.
    """
    import os
    import requests

    if token is None:
        token = os.environ.get("GITHUB_TOKEN", None)
    if token is None and token_required:
        raise ValueError("GitHub token is required")
    if use_pygithub is None:
        from vectorbtpro.utils.module_ import check_installed

        use_pygithub = check_installed("github")

    if use_pygithub:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("github")
        from github import Github, Auth
        from github.GithubException import UnknownObjectException

        if token is not None:
            g = Github(auth=Auth.Token(token))
        else:
            g = Github()
        try:
            repo = g.get_repo(f"{repo_owner}/{repo_name}")
        except (requests.RequestException, UnknownObjectException):
            if not raise_missing:
                return None
            raise Exception(f"Repository '{repo_owner}/{repo_name}' not found or access denied")
        releases = repo.get_releases()
        found_release = None
        for release in releases:
            if release.title == release_name:
                found_release = release
        if found_release is None:
            if not raise_missing:
                return None
            raise Exception(f"Release {release_name!r} not found")
        return resolve_asset_name_from_names(
            [get_asset_value(asset, "name") for asset in found_release.get_assets()],
            asset_name=asset_name,
            release_name=release_name,
            raise_missing=raise_missing,
        )
    else:
        headers = {"Accept": "application/vnd.github+json"}
        if token is not None:
            headers["Authorization"] = f"token {token}"
        releases_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
        try:
            response = requests.get(releases_url, headers=headers)
            response.raise_for_status()
        except requests.RequestException:
            if not raise_missing:
                return None
            raise
        releases = response.json()
        release_info = None
        for release in releases:
            if release.get("name") == release_name:
                release_info = release
        if release_info is None:
            if not raise_missing:
                return None
            raise Exception(f"Release {release_name!r} not found")
        return resolve_asset_name_from_names(
            [get_asset_value(asset, "name") for asset in release_info.get("assets", [])],
            asset_name=asset_name,
            release_name=release_name,
            raise_missing=raise_missing,
        )


def download_github_asset(
    repo_owner: str,
    repo_name: str,
    release_name: str,
    asset_name: str,
    raise_missing: bool = True,
    token: tp.Optional[str] = None,
    token_required: bool = False,
    use_pygithub: tp.Optional[bool] = None,
    chunk_size: int = 8192,
    cache_dir: tp.Optional[tp.PathLike] = None,
    cache_mkdir_kwargs: tp.KwargsLike = None,
    clear_cache: bool = False,
    show_progress: bool = True,
    pbar_kwargs: tp.KwargsLike = None,
) -> tp.Union[bytes, Path]:
    """Download an asset from a GitHub release.

    Args:
        repo_owner (str): Owner of the GitHub repository.
        repo_name (str): Name of the GitHub repository.
        release_name (str): GitHub release tag (e.g. 'v2024.12.15').

            Must be a concrete release name. Use `resolve_release_name`.
        asset_name (str): Name of the asset file (e.g. 'messages.json').

            Must be a concrete asset name. Use `resolve_asset_name`.
        raise_missing (bool): Whether to raise if no matching release or asset is found.
        token (Optional[str]): GitHub authentication token.
        token_required (bool): Flag indicating whether a GitHub token is required.
        use_pygithub (Optional[bool]): Use the PyGithub library to fetch release data.

            If True, uses https://github.com/PyGithub/PyGithub (otherwise requests).
        chunk_size (int): Number of bytes per download chunk.
        cache_dir (Optional[PathLike]): Directory for caching downloaded files.
        cache_mkdir_kwargs (KwargsLike): Keyword arguments for cache directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        clear_cache (bool): Remove cached file before downloading if True.
        show_progress (bool): Flag indicating whether to display the progress bar.
        pbar_kwargs (KwargsLike): Keyword arguments for configuring the progress bar.

            See `vectorbtpro.utils.pbar.ProgressBar`.

    Returns:
        Union[bytes, Path]: Path to cached file if `cache_dir` is set,
            otherwise raw bytes of the downloaded asset.
    """
    import os

    import requests

    if token is None:
        token = os.environ.get("GITHUB_TOKEN", None)
    if token is None and token_required:
        raise ValueError("GitHub token is required")
    if use_pygithub is None:
        from vectorbtpro.utils.module_ import check_installed

        use_pygithub = check_installed("github")
    if cache_mkdir_kwargs is None:
        cache_mkdir_kwargs = {}
    if pbar_kwargs is None:
        pbar_kwargs = {}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)

    if use_pygithub:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("github")
        from github import Github, Auth
        from github.GithubException import UnknownObjectException

        if token is not None:
            g = Github(auth=Auth.Token(token))
        else:
            g = Github()
        try:
            repo = g.get_repo(f"{repo_owner}/{repo_name}")
        except (requests.RequestException, UnknownObjectException):
            if not raise_missing:
                return None
            raise Exception(f"Repository '{repo_owner}/{repo_name}' not found or access denied")
        releases = repo.get_releases()
        found_release = None
        for release in releases:
            if release.title == release_name:
                found_release = release
        if found_release is None:
            if not raise_missing:
                return None
            raise Exception(f"Release {release_name!r} not found")
        release = found_release
        assets = list(release.get_assets())
        asset = next((a for a in assets if get_asset_value(a, "name") == asset_name), None)
        if asset is None:
            if not raise_missing:
                return None
            raise Exception(f"Asset {asset_name!r} not found in release {release_name!r}")
        asset_url = get_asset_value(asset, "url")
    else:
        headers = {"Accept": "application/vnd.github+json"}
        if token is not None:
            headers["Authorization"] = f"token {token}"
        releases_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
        try:
            response = requests.get(releases_url, headers=headers)
            response.raise_for_status()
        except requests.RequestException:
            if not raise_missing:
                return None
            raise
        releases = response.json()
        release_info = None
        for release in releases:
            if release.get("name") == release_name:
                release_info = release
        if release_info is None:
            if not raise_missing:
                return None
            raise Exception(f"Release {release_name!r} not found")
        assets = release_info.get("assets", [])
        asset = next((a for a in assets if get_asset_value(a, "name") == asset_name), None)
        if asset is None:
            if not raise_missing:
                return None
            raise Exception(f"Asset {asset_name!r} not found in release {release_name!r}")
        asset_url = get_asset_value(asset, "url")

    actual_asset_name = get_asset_value(asset, "name")
    if cache_dir is not None and clear_cache:
        cache_file = cache_dir / actual_asset_name
        if cache_file.exists():
            cache_file.unlink()

    asset_headers = {"Accept": "application/octet-stream"}
    if token is not None:
        asset_headers["Authorization"] = f"token {token}"
    try:
        asset_response = requests.get(asset_url, headers=asset_headers, stream=True)
        asset_response.raise_for_status()
    except requests.RequestException:
        if not raise_missing:
            return None
        raise
    file_size = int(asset_response.headers.get("Content-Length", 0))
    if file_size == 0:
        file_size = get_asset_value(asset, "size")
    if show_progress is None:
        show_progress = True
    pbar_kwargs = flat_merge_dicts(
        dict(
            bar_id=get_caller_qualname(),
            unit="iB",
            unit_scale=True,
            prefix=f"Downloading {actual_asset_name}",
        ),
        pbar_kwargs,
    )

    if cache_dir is not None:
        check_mkdir(cache_dir, **cache_mkdir_kwargs)
        cache_file = cache_dir / actual_asset_name
        with open(cache_file, "wb") as f:
            with ProgressBar(total=file_size, show_progress=show_progress, **pbar_kwargs) as pbar:
                for chunk in asset_response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return cache_file
    else:
        with io.BytesIO() as bytes_io:
            with ProgressBar(total=file_size, show_progress=show_progress, **pbar_kwargs) as pbar:
                for chunk in asset_response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        bytes_io.write(chunk)
                        pbar.update(len(chunk))
            return bytes_io.getvalue()
