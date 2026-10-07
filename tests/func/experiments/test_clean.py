import logging
import os

import pytest

from dvc.repo.experiments.executor.base import ExecutorInfo, TaskStatus
from dvc.repo.experiments.workspaces import normpath

REV = "a" * 40
CLEAN_LOGGER = "dvc.repo.experiments.clean"


def _dump_lockfile(path, outs):
    import yaml

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fobj:
        yaml.safe_dump({"stages": {"orphan": {"outs": outs}}}, fobj)


def _make_executor_infofile(dvc, scm, queue, rev, root_dir, name):
    infofile = queue.get_infofile_path(rev)
    root = os.path.abspath(root_dir)
    info = ExecutorInfo(
        git_url=root.replace(os.sep, "/"),
        baseline_rev=scm.get_rev(),
        location="tempdir",
        root_dir=root,
        dvc_dir=".dvc",
        name=name,
        status=TaskStatus.RUNNING,
    )
    info.dump_json(infofile)
    return infofile


def test_exp_clean_keeps_output_referenced_by_workspace(tmp_dir, scm, dvc, caplog):
    """Shared output referenced by the main workspace must not be removed."""
    tmp_dir.scm_gen("README", "init", commit="init")

    shared = tmp_dir / "shared" / "data"
    shared.write_text("survive")
    dvc.stage.add(name="shared-stage", cmd="true", outs=["shared/data"])

    queue = dvc.experiments.tempdir_queue
    exp_root = os.path.join(queue._standalone_tmp_dir, "tmpORPHAN")
    os.makedirs(exp_root, exist_ok=True)
    internal = os.path.join(exp_root, "internal.txt")
    with open(internal, "w", encoding="utf-8") as fobj:
        fobj.write("orphan-only")
    _dump_lockfile(
        os.path.join(exp_root, "dvc.lock"),
        [
            {"path": str(shared), "md5": "0" * 32, "size": 7},
            {"path": "internal.txt", "md5": "1" * 32, "size": 9},
        ],
    )
    infofile = _make_executor_infofile(dvc, scm, queue, REV, exp_root, "orphan-exp")

    with caplog.at_level(logging.WARNING, logger=CLEAN_LOGGER):
        report = dvc.experiments.clean()

    # orphan workspace and its stale executor info are reclaimed
    assert not os.path.exists(exp_root)
    assert not os.path.exists(os.path.dirname(infofile))
    # shared output is preserved and untouched
    assert shared.exists()
    assert shared.read_text() == "survive"
    # the mismatch is reported as a replaced output, not silently removed
    shared_abs = normpath(str(shared))
    assert shared_abs in [path for _, path, _, _ in report.replaced]
    assert shared_abs in report.protected
    assert "was replaced" in caplog.text


def test_exp_clean_reports_missing_workspace(tmp_dir, scm, dvc, caplog):
    """An interrupted executor whose workspace is gone is reported separately."""
    tmp_dir.scm_gen("README", "init", commit="init")

    queue = dvc.experiments.tempdir_queue
    missing_root = os.path.join(queue._standalone_tmp_dir, "tmpMISSING")
    infofile = _make_executor_infofile(
        dvc, scm, queue, REV, missing_root, "interrupted-exp"
    )

    with caplog.at_level(logging.WARNING, logger=CLEAN_LOGGER):
        report = dvc.experiments.clean()

    assert report.missing_workspaces == ["interrupted-exp"]
    assert not os.path.exists(os.path.dirname(infofile))
    assert "no longer exists" in caplog.text


def test_exp_clean_normal_flow_unchanged(tmp_dir, scm, dvc, mocker):
    """With no leftover workspaces, only dvc-task messages are cleaned."""
    clean = mocker.spy(dvc.experiments.celery_queue.celery, "clean")

    report = dvc.experiments.clean()

    clean.assert_called_once_with()
    assert not report.issues
    assert not report.protected
    assert not report.removed_workspaces


@pytest.mark.parametrize("pid_exists", [False, True])
def test_exp_clean_keeps_running_executor(tmp_dir, scm, dvc, mocker, pid_exists):
    """Executors with a live process are never treated as orphans."""
    tmp_dir.scm_gen("README", "init", commit="init")

    queue = dvc.experiments.tempdir_queue
    exp_root = os.path.join(queue._standalone_tmp_dir, "tmpRUNNING")
    os.makedirs(exp_root, exist_ok=True)
    _dump_lockfile(
        os.path.join(exp_root, "dvc.lock"),
        [{"path": "internal.txt", "md5": "1" * 32}],
    )
    internal = os.path.join(exp_root, "internal.txt")
    with open(internal, "w", encoding="utf-8") as fobj:
        fobj.write("running")
    infofile = _make_executor_infofile(dvc, scm, queue, REV, exp_root, "running-exp")

    from dvc_task.proc.process import ProcessInfo

    mocker.patch.object(
        ProcessInfo, "load", return_value=mocker.Mock(pid=123 if pid_exists else None)
    )
    mocker.patch("psutil.pid_exists", return_value=pid_exists)

    dvc.experiments.clean()

    if pid_exists:
        assert os.path.isdir(exp_root)
        assert os.path.isfile(infofile)
    else:
        assert not os.path.exists(exp_root)
        assert not os.path.exists(os.path.dirname(infofile))
