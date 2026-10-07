import os
from typing import TYPE_CHECKING, Optional

from dvc.log import logger
from dvc.ui import ui
from dvc.utils import relpath
from dvc.utils.fs import remove

from .executor.base import ExecutorInfo, TaskStatus
from .workspaces import (
    CleanupReport,
    LockedOutput,
    WorkspaceOutputs,
    cleanup_workspaces,
    load_workspace,
    normpath,
)

if TYPE_CHECKING:
    from dvc.repo import Repo

WORKSPACE_NAME = "workspace"

logger = logger.getChild(__name__)


def clean(repo: "Repo") -> CleanupReport:
    """Clean up dvc-task messages and orphaned experiment workspaces.

    Outputs in shared locations are only removed when the experiment's own
    lock record matches the file on disk and no surviving experiment still
    references the same path.
    """
    ui.write("Cleaning up dvc-task messages...")
    # Broker state must be inspected before celery messages are purged.
    referenced = _broker_referenced_revs(repo)
    workspaces, targets, stale_infofiles = _collect_workspaces(repo, referenced)
    report = cleanup_workspaces(workspaces, targets)

    repo.experiments.celery_queue.celery.clean()

    for infofile in stale_infofiles:
        remove(os.path.dirname(infofile))

    _log_report(report)
    ui.write("Done!")
    return report


def _broker_referenced_revs(repo: "Repo") -> set[str]:
    """Stash revisions still tracked by the celery task broker.

    Their executor workspaces may be redelivered and restarted and must never
    be reclaimed.
    """
    queue = repo.experiments.celery_queue
    revs: set[str] = set()
    try:
        for _, entry in queue._iter_queued():
            revs.add(entry.stash_rev)
        for _, entry in queue._iter_processed():
            revs.add(entry.stash_rev)
    except Exception:  # noqa: BLE001 - broker state is best-effort here
        logger.debug("Unable to inspect experiment task broker", exc_info=True)
    return revs


def _process_alive(queue, name: str) -> bool:
    """Check whether the executor process for ``name`` is still running."""
    import psutil

    from dvc_task.proc.process import ProcessInfo

    info_path = os.path.join(queue.pid_dir, name, f"{name}.json")
    try:
        proc_info = ProcessInfo.load(info_path)
    except (OSError, ValueError):
        return False
    if proc_info.pid is None:
        return False
    if not psutil.pid_exists(proc_info.pid):
        logger.debug("Executor PID '%d' no longer exists.", proc_info.pid)
        return False
    return True


def _iter_infofiles(repo: "Repo"):
    seen: set[str] = set()
    for queue in (
        repo.experiments.celery_queue,
        repo.experiments.tempdir_queue,
    ):
        pid_dir = queue.pid_dir
        if not os.path.isdir(pid_dir):
            continue
        for name in os.listdir(pid_dir):
            infofile = queue.get_infofile_path(name)
            if infofile in seen or not os.path.isfile(infofile):
                continue
            seen.add(infofile)
            try:
                info = ExecutorInfo.load_json(infofile)
            except (OSError, ValueError):
                logger.debug("Skipping unreadable executor info '%s'", infofile)
                continue
            yield name, info, infofile, _process_alive(queue, name)


def _outputs_from_repo(dvc: "Repo", name: str, root_dir: str) -> WorkspaceOutputs:
    outputs: dict[str, LockedOutput] = {}
    for out in dvc.index.outs:
        if out.protocol != "local":
            continue
        hash_info = out.hash_info
        outputs[normpath(out.fs_path)] = LockedOutput(
            path=normpath(out.fs_path),
            checksum=hash_info.value if hash_info else None,
            use_cache=out.use_cache,
        )
    return WorkspaceOutputs(
        name=name,
        root_dir=os.path.abspath(root_dir),
        outputs=outputs,
        exists=True,
    )


def _open_workspace(name: str, info: ExecutorInfo) -> WorkspaceOutputs:
    """Build a workspace record from the executor workspace's own repo/lock."""
    root_dir = os.path.abspath(info.root_dir)
    if not os.path.isdir(root_dir):
        return load_workspace(name, root_dir)

    dvc_root = os.path.join(root_dir, info.dvc_dir)
    try:
        from dvc.repo import Repo

        with Repo(dvc_root) as dvc:
            return _outputs_from_repo(dvc, name, root_dir)
    except Exception:  # noqa: BLE001 - fall back to the raw lock record
        logger.debug(
            "Unable to open experiment workspace '%s', reading lock file",
            root_dir,
            exc_info=True,
        )
        return load_workspace(name, root_dir)


def _collect_workspaces(
    repo: "Repo", referenced: set[str]
) -> tuple[list[WorkspaceOutputs], list[str], list[str]]:
    """Return (survivors, cleanup targets, stale infofile dirs).

    Surviving workspaces are the current workspace plus any experiment that
    is still running or queued. Dead executors which the broker no longer
    references are treated as orphaned cleanup targets.
    """
    main = _outputs_from_repo(repo, WORKSPACE_NAME, repo.scm.root_dir)
    workspaces: list[WorkspaceOutputs] = [main]
    targets: list[str] = []
    stale_infofiles: list[str] = []
    used_names = {WORKSPACE_NAME}

    def unique_name(info: ExecutorInfo, rev: str) -> str:
        name = info.name or rev[:7]
        if name not in used_names:
            used_names.add(name)
            return name
        name = f"{name}-{rev[:7]}"
        used_names.add(name)
        return name

    for name, info, infofile, alive in _iter_infofiles(repo):
        if name == WORKSPACE_NAME:
            # The in-place workspace executor is the main workspace itself.
            continue

        display_name = unique_name(info, name)
        root_exists = os.path.isdir(info.root_dir)

        if alive or name in referenced:
            # Still running, or the broker may redeliver and restart it.
            if root_exists:
                workspaces.append(_open_workspace(display_name, info))
            continue

        if not root_exists:
            interrupted = info.status <= TaskStatus.SUCCESS
            workspace = load_workspace(display_name, info.root_dir)
            if interrupted:
                workspaces.append(workspace)
                targets.append(workspace.name)
            stale_infofiles.append(infofile)
            continue

        workspace = _open_workspace(display_name, info)
        workspaces.append(workspace)
        targets.append(workspace.name)
        stale_infofiles.append(infofile)

    return workspaces, targets, stale_infofiles


def _short(checksum: Optional[str]) -> str:
    return checksum[:7] if checksum else "unknown"


def _log_report(report: CleanupReport) -> None:
    for name in report.missing_workspaces:
        logger.warning("Experiment workspace '%s' no longer exists.", name)
    for name, path in report.broken_references:
        logger.warning(
            "Output '%s' referenced by experiment '%s' is missing.",
            relpath(path),
            name,
        )
    for name, path, expected, actual in report.replaced:
        logger.warning(
            "Output '%s' referenced by experiment '%s' was replaced "
            "(expected '%s', found '%s'); not removing it.",
            relpath(path),
            name,
            _short(expected),
            _short(actual),
        )
    for path in report.protected:
        logger.info(
            "Keeping '%s', it is still referenced by another experiment.",
            relpath(path),
        )
    for path in report.removed_files:
        logger.debug("Removed experiment output '%s'", relpath(path))
    for path in report.removed_workspaces:
        logger.debug("Removed experiment workspace '%s'", relpath(path))
