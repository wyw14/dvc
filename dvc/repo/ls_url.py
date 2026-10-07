from typing import Any

from fsspec.implementations.local import LocalFileSystem as _LocalFileSystem

from dvc.exceptions import URLMissingError
from dvc.fs import LocalFileSystem, parse_external_url
from dvc.fs.paginate import Entries, PaginatedWalk
from dvc.log import logger

logger = logger.getChild(__name__)


def _walk_entries(fs, fs_path, recursive, maxdepth):
    """Collect ``{relative path: entry}`` from a (possibly paginated) walk.

    Returns ``(entries, incomplete, error)``. For paginated remotes, entries
    from confirmed levels are retained when the listing cannot be completed.
    """
    if isinstance(fs, LocalFileSystem):
        # dvc's LocalFileSystem does not support maxdepth yet
        walker = _LocalFileSystem().walk(fs_path, detail=True, maxdepth=maxdepth)
        level_walk = None
    else:
        # Traverse the remote level by level, following pagination tokens and
        # retaining confirmed entries when the listing cannot be completed.
        level_walk = PaginatedWalk(
            fs, fs_path, maxdepth=maxdepth if recursive else None
        )
        walker = iter(level_walk)

    # De-duplicate by relative path: paginated remotes may repeat an object
    # name on different pages. Directory markers and files stay distinguishable
    # via the entry's "type" until the output entries are built below.
    entries: dict[str, dict[str, Any]] = {}
    for root, dirs, files in walker:
        parts = fs.relparts(root, fs_path)
        if parts == (".",):
            parts = ()
        if not recursive or (maxdepth and len(parts) >= maxdepth - 1):
            files.update(dirs)

        for info in files.values():
            ls_info = {
                "path": fs.relpath(info["name"], fs_path),
                "isdir": info["type"] == "directory",
                "size": info.get("size"),
            }
            entries.setdefault(ls_info["path"], ls_info)

        if not recursive:
            break

    if level_walk is None:
        return entries, False, None
    return entries, not level_walk.complete, level_walk.error


def ls_url(url, *, fs_config=None, recursive=False, maxdepth=None, config=None):
    fs, fs_path = parse_external_url(url, fs_config=fs_config, config=config)
    try:
        info = fs.info(fs_path)
    except FileNotFoundError as exc:
        raise URLMissingError(url) from exc
    if maxdepth == 0 or info["type"] != "directory":
        return Entries([{"path": info["name"], "isdir": False}])

    entries, incomplete, error = _walk_entries(fs, fs_path, recursive, maxdepth)
    if incomplete:
        logger.warning(
            "Listing of '%s' is incomplete (%s); %d confirmed entr%s "
            "returned, the remote may contain more",
            url,
            error,
            len(entries),
            "y" if len(entries) == 1 else "ies",
        )

    return Entries(entries.values(), incomplete=incomplete, error=error)
