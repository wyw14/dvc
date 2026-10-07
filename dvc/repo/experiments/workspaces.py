"""Reconcile outputs across experiment workspaces.

Multiple experiment workspaces may reference outputs that live in the same
physical directory (for example external outputs, or several experiments
sharing one working directory). Each workspace keeps its own lock record
(``dvc.lock``) that describes the outputs it references and their expected
checksums.

When one experiment is cleaned up (or switched away), an output file may only
be removed when both of the following are true:

1. The file on disk matches that experiment's own lock record (it really is
   this experiment's output and was not replaced by another experiment).
2. No other, surviving, experiment's lock record references the same path.

Three kinds of issues are reported separately:

* ``broken_references``: an output referenced by a lock record no longer
  exists (the reference is dangling).
* ``replaced``: an output path still exists, but its checksum no longer
  matches the lock record (it has been replaced by another experiment).
* ``missing_workspaces``: the experiment workspace directory itself is gone.

Workspaces that contain a single experiment, and flows that do not share
outputs, keep their original behavior (the entire workspace directory is
simply removed).
"""

import hashlib
import logging
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

try:  # pragma: no cover - exercised implicitly
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

LOCKFILE_NAME = "dvc.lock"
#: Lock record keys which carry the expected checksum of an output.
_CHECKSUM_KEYS = ("md5", "checksum", "etag", "hash")
#: Output checksum suffix that marks directory outputs.
_DIR_SUFFIX = ".dir"

Hasher = Callable[[str], Optional[str]]
Remover = Callable[[str], None]
BrokenReference = tuple[str, str]
ReplacedOutput = tuple[str, str, Optional[str], Optional[str]]


@dataclass(frozen=True)
class LockedOutput:
    """A single output referenced by an experiment lock record."""

    path: str
    checksum: Optional[str]
    use_cache: bool = True

    @property
    def is_dir(self) -> bool:
        return bool(self.checksum and self.checksum.endswith(_DIR_SUFFIX))


@dataclass
class WorkspaceOutputs:
    """Outputs that one experiment workspace references via its own lock."""

    name: str
    root_dir: str
    outputs: dict[str, LockedOutput] = field(default_factory=dict)
    exists: bool = True

    def shared_paths(self, other: "WorkspaceOutputs") -> set[str]:
        """Return output paths referenced by both workspaces."""
        return self.outputs.keys() & other.outputs.keys()


@dataclass
class CleanupReport:
    """Result of reconciling and cleaning experiment workspaces."""

    removed_files: list[str] = field(default_factory=list)
    removed_workspaces: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    broken_references: list[BrokenReference] = field(default_factory=list)
    replaced: list[ReplacedOutput] = field(default_factory=list)
    missing_workspaces: list[str] = field(default_factory=list)

    def update(self, other: "CleanupReport") -> None:
        self.removed_files.extend(other.removed_files)
        self.removed_workspaces.extend(other.removed_workspaces)
        self.protected.extend(other.protected)
        self.broken_references.extend(other.broken_references)
        self.replaced.extend(other.replaced)
        self.missing_workspaces.extend(other.missing_workspaces)

    @property
    def issues(self) -> bool:
        return bool(self.broken_references or self.replaced or self.missing_workspaces)


def normpath(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _default_remove(path: str) -> None:
    """Remove a file or directory, matching DVC's remove behavior."""

    def on_error(func, p, _excinfo):
        mode = os.lstat(p).st_mode
        os.chmod(p, mode | stat.S_IWRITE)
        func(p)

    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, onerror=on_error)
        else:
            os.unlink(path)
    except FileNotFoundError:
        pass


def local_file_checksum(path: str) -> Optional[str]:
    """Return the checksum of a local file.

    Uses ``dvc-objects`` when available so that the value matches what DVC
    stores in lock records, and falls back to a plain content md5 otherwise.
    """
    try:
        from dvc_objects.fs import localfs

        checksum = localfs.checksum(path)
        if checksum:
            return str(checksum)
    except Exception:  # noqa: BLE001 - checksums are best-effort
        logger.debug("Unable to compute checksum for '%s'", path, exc_info=True)

    hasher = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fobj:
        for chunk in iter(lambda: fobj.read(2**20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_lockfile(path: str) -> Mapping[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read lock files")
    import yaml as _yaml

    with open(path, encoding="utf-8") as fobj:
        data = _yaml.safe_load(fobj) or {}
    return data if isinstance(data, Mapping) else {}


def parse_locked_outputs(
    lockfile: str,
    base_dir: Optional[str] = None,
) -> dict[str, LockedOutput]:
    """Parse a ``dvc.lock`` file into absolute-path output records.

    ``wdir`` is honored for stages that run from a subdirectory. Paths are
    returned as normalized absolute paths.
    """
    base_dir = os.path.abspath(base_dir or os.path.dirname(lockfile))
    data = _read_lockfile(lockfile)
    stages = data.get("stages") or {}

    outputs: dict[str, LockedOutput] = {}
    for stage in stages.values():
        if not isinstance(stage, Mapping):
            continue
        wdir = os.path.abspath(os.path.join(base_dir, stage.get("wdir") or ""))
        for out in stage.get("outs") or []:
            if not isinstance(out, Mapping):
                continue
            raw_path = out.get("path")
            if not raw_path:
                continue
            out_path = normpath(os.path.join(wdir, raw_path))
            checksum = next((out[key] for key in _CHECKSUM_KEYS if out.get(key)), None)
            outputs[out_path] = LockedOutput(
                path=out_path,
                checksum=str(checksum) if checksum is not None else None,
                use_cache=bool(out.get("cache", True)),
            )
    return outputs


def load_workspace(
    name: str,
    root_dir: str,
    lockfile: Optional[str] = None,
    outputs: Optional[Mapping[str, LockedOutput]] = None,
) -> WorkspaceOutputs:
    """Build a workspace record from its root directory and lock record.

    Outputs may be supplied directly (for example, from an open DVC repo
    index); otherwise ``<root_dir>/dvc.lock`` is parsed.
    """
    root = os.path.abspath(root_dir)
    exists = os.path.isdir(root)
    if outputs is not None:
        locked = dict(outputs)
    elif exists:
        lockfile = lockfile or os.path.join(root, LOCKFILE_NAME)
        locked = (
            parse_locked_outputs(lockfile, root) if os.path.isfile(lockfile) else {}
        )
    else:
        locked = {}
    return WorkspaceOutputs(name=name, root_dir=root, outputs=locked, exists=exists)


def _classify_output(
    name: str,
    out_path: str,
    out: LockedOutput,
    hasher: Hasher,
) -> tuple[Optional[BrokenReference], Optional[ReplacedOutput]]:
    if not os.path.lexists(out_path):
        return (name, out_path), None
    if not out.checksum:
        # Outputs without checksums cannot be ownership-checked by hash.
        return None, None

    mismatch: Optional[ReplacedOutput] = None
    if out.is_dir:
        if not os.path.isdir(out_path):
            mismatch = (name, out_path, out.checksum, None)
    elif os.path.isdir(out_path):
        mismatch = (name, out_path, out.checksum, None)
    else:
        actual = _safe_checksum(hasher, out_path)
        if actual is None:
            return (name, out_path), None
        if actual != out.checksum:
            mismatch = (name, out_path, out.checksum, actual)
    return None, mismatch


def _safe_checksum(hasher: Hasher, path: str) -> Optional[str]:
    try:
        return hasher(path)
    except OSError:
        return None


def classify_outputs(
    workspace: WorkspaceOutputs,
    hasher: Hasher = local_file_checksum,
    paths: Optional["Iterable[str]"] = None,
) -> tuple[list[BrokenReference], list[ReplacedOutput]]:
    """Classify the state of a workspace's locked outputs.

    Args:
        workspace: Workspace whose locked outputs are checked.
        hasher: Callable returning the on-disk checksum of a file.
        paths: Optional restriction to these (normalized) output paths.

    Returns ``(broken_references, replaced)`` where:

    * a broken reference is a locked output path that no longer exists;
    * a replaced output exists on disk but its checksum does not match the
      lock record, reported as ``(workspace, path, expected, actual)``.
    """
    broken: list[BrokenReference] = []
    replaced: list[ReplacedOutput] = []
    if not workspace.exists:
        return broken, replaced

    restricted = None if paths is None else {normpath(p) for p in paths}
    for out_path, out in workspace.outputs.items():
        if restricted is not None and out_path not in restricted:
            continue
        broken_ref, replaced_out = _classify_output(
            workspace.name, out_path, out, hasher
        )
        if broken_ref:
            broken.append(broken_ref)
        if replaced_out:
            replaced.append(replaced_out)
    return broken, replaced


def _prune_workspace(
    workspace: WorkspaceOutputs,
    keep: set[str],
    remover: Remover,
) -> bool:
    """Remove workspace contents except protected paths.

    Returns True if the workspace root directory itself was removed.
    """
    root = normpath(workspace.root_dir)

    def is_protected(path: str) -> bool:
        return path in keep or any(
            path.startswith(protected + os.sep) for protected in keep
        )

    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for filename in filenames:
            path = normpath(os.path.join(dirpath, filename))
            if not is_protected(path):
                remover(path)
        for dirname in dirnames:
            path = normpath(os.path.join(dirpath, dirname))
            if (
                not is_protected(path)
                and os.path.isdir(path)
                and not os.path.islink(path)
            ):
                remover(path)
    try:
        os.rmdir(root)
        return True
    except OSError:
        return False


def _is_inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def cleanup_workspace(  # noqa: C901, PLR0912
    workspace: WorkspaceOutputs,
    survivors: "Iterable[WorkspaceOutputs]" = (),
    hasher: Hasher = local_file_checksum,
    remover: Remover = _default_remove,
) -> CleanupReport:
    """Clean up one workspace, protecting outputs of surviving workspaces.

    Output ownership is decided solely by the workspace's own lock record:
    a file is only removed if it matches the lock record and is not still
    referenced by another workspace.
    """
    report = CleanupReport()
    root = normpath(workspace.root_dir)

    if not workspace.exists:
        report.missing_workspaces.append(workspace.name)
        return report

    protected: set[str] = set()
    shared_root = False
    for other in survivors:
        if other.exists and normpath(other.root_dir) == root:
            shared_root = True
        protected.update(other.outputs)

    # Internal outputs of a private workspace directory are removed together
    # with the directory (the original single-experiment behavior). Only
    # shared locations can affect other experiments and need reconciliation.
    if shared_root:
        shared_paths = set(workspace.outputs)
    else:
        shared_paths = {
            path for path in workspace.outputs if not _is_inside(path, root)
        }

    broken, replaced = classify_outputs(workspace, hasher=hasher, paths=shared_paths)
    report.broken_references.extend(broken)
    report.replaced.extend(replaced)
    replaced_paths = {path for _, path, _, _ in replaced}

    removable: list[str] = []
    for path in shared_paths:
        if path in protected:
            # Still referenced by another experiment, do not move it.
            if os.path.lexists(path):
                report.protected.append(path)
            continue
        if not os.path.lexists(path) or path in replaced_paths:
            # Dangling reference, or the file now belongs to another
            # experiment; either way it must not be moved.
            continue
        removable.append(path)

    if shared_root:
        # Several experiments share this directory. Only remove the files
        # that this experiment's own lock record claims as its own.
        for path in removable:
            remover(path)
            report.removed_files.append(path)
        return report

    external = [path for path in removable if not _is_inside(path, root)]
    protected_inside = {path for path in protected if _is_inside(path, root)}
    if protected_inside:
        # One or more shared outputs live inside this workspace directory;
        # prune everything else instead of removing the whole directory.
        if _prune_workspace(workspace, protected_inside, remover):
            report.removed_workspaces.append(workspace.root_dir)
    else:
        # Single-experiment directory without shared in-tree outputs:
        # preserve the original behavior and remove it wholesale.
        remover(workspace.root_dir)
        report.removed_workspaces.append(workspace.root_dir)

    # Outputs in a shared/external directory are removed individually.
    for path in external:
        remover(path)
        report.removed_files.append(path)
    return report


def cleanup_workspaces(
    workspaces: "Iterable[WorkspaceOutputs]",
    targets: "Iterable[str]",
    hasher: Hasher = local_file_checksum,
    remover: Remover = _default_remove,
) -> CleanupReport:
    """Clean up the named target workspaces.

    All other workspaces are treated as survivors whose locked output
    references must be preserved.
    """
    workspaces = list(workspaces)
    target_names = set(targets)
    targets_by_name = {ws.name: ws for ws in workspaces if ws.name in target_names}

    missing = target_names - set(targets_by_name)
    if missing:
        raise KeyError(sorted(missing))

    report = CleanupReport()
    for name in sorted(target_names):
        workspace = targets_by_name[name]
        survivors = (
            ws
            for ws in workspaces
            if ws is not workspace and ws.name not in target_names
        )
        report.update(
            cleanup_workspace(
                workspace,
                survivors=survivors,
                hasher=hasher,
                remover=remover,
            )
        )
    return report
