import os

import pytest

from dvc.repo.experiments.workspaces import (
    LockedOutput,
    WorkspaceOutputs,
    cleanup_workspaces,
    load_workspace,
    normpath,
    parse_locked_outputs,
)


def _hasher(path):
    # Files used in these tests are written with their checksum as content.
    with open(path, encoding="utf-8") as fobj:
        return fobj.read().strip()


def _write(path, content=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fobj:
        fobj.write(content)


def _workspace(name, root, **outs):
    outputs = {
        normpath(path): LockedOutput(normpath(path), checksum)
        for path, checksum in outs.items()
    }
    exists = os.path.isdir(root)
    return WorkspaceOutputs(name, root, outputs, exists=exists)


@pytest.fixture
def layout(tmp_path):
    shared = tmp_path / "shared"
    root_a = tmp_path / "wa"
    root_b = tmp_path / "wb"
    ext = shared / "ext.dat"

    _write(str(ext), "hashB")
    _write(str(root_a / "a" / "model"), "A-model")
    _write(str(root_a / "junk"), "junk")
    _write(str(root_b / "b" / "model"), "B-model")

    ws_a = _workspace(
        "A",
        str(root_a),
        **{str(ext): "hashA", str(root_a / "a" / "model"): "A-model"},
    )
    ws_b = _workspace(
        "B",
        str(root_b),
        **{str(ext): "hashB", str(root_b / "b" / "model"): "B-model"},
    )
    return ws_a, ws_b, ext, root_a, root_b


def test_single_workspace_is_removed_wholesale(tmp_path):
    root = tmp_path / "w"
    out = root / "model"
    _write(str(out), "v1")
    _write(str(root / "untracked"), "x")
    workspace = _workspace("S", str(root), **{str(out): "v1"})

    report = cleanup_workspaces([workspace], ["S"], hasher=_hasher)

    assert not root.exists()
    assert report.removed_workspaces == [str(root)]
    assert not report.issues
    assert not report.protected


def test_shared_external_output_is_not_removed(layout):
    ws_a, ws_b, ext, root_a, root_b = layout

    report = cleanup_workspaces([ws_a, ws_b], ["A"], hasher=_hasher)

    assert not root_a.exists()
    assert root_b.is_dir()
    assert ext.exists()
    assert ext.read_text() == "hashB"
    assert report.removed_workspaces == [str(root_a)]
    assert report.protected == [normpath(str(ext))]
    assert [path for _, path, _, _ in report.replaced] == [normpath(str(ext))]
    assert not report.broken_references
    assert not report.missing_workspaces


def test_shared_root_is_pruned_selectively(tmp_path):
    root = tmp_path / "shared-root"
    out_a = root / "a.out"
    out_b = root / "b.out"
    _write(str(out_a), "A")
    _write(str(out_b), "B")
    ws_a = _workspace("A", str(root), **{str(out_a): "A"})
    ws_b = _workspace("B", str(root), **{str(out_b): "B"})

    report = cleanup_workspaces([ws_a, ws_b], ["A"], hasher=_hasher)

    assert root.is_dir()
    assert not out_a.exists()
    assert out_b.read_text() == "B"
    assert not report.removed_workspaces


def test_modified_internal_outputs_do_not_block_wholesale_removal(tmp_path):
    # Normal single-experiment flow: the on-disk output no longer matches
    # the lock record, but since it lives in the private workspace directory
    # it is removed together with the directory and no issue is reported.
    root = tmp_path / "w"
    out = root / "model"
    _write(str(out), "new-content")
    workspace = _workspace("S", str(root), **{str(out): "old-checksum"})

    report = cleanup_workspaces([workspace], ["S"], hasher=_hasher)

    assert not root.exists()
    assert not report.issues
    assert not report.protected


def test_broken_reference_is_reported_and_not_removed(tmp_path):
    root = tmp_path / "w"
    gone = tmp_path / "shared" / "gone.dat"
    _write(str(root / "x"), "x")
    workspace = _workspace("Y", str(root), **{str(gone): "g"})

    report = cleanup_workspaces([workspace], ["Y"], hasher=_hasher)

    assert report.broken_references == [("Y", normpath(str(gone)))]
    assert not report.removed_files
    assert not root.exists()


def test_replaced_shared_output_is_not_moved(tmp_path):
    root = tmp_path / "w"
    shared = tmp_path / "shared"
    out = shared / "out"
    _write(str(out), "NEW")
    _write(str(root / "x"), "x")
    workspace = _workspace("R", str(root), **{str(out): "OLD"})

    report = cleanup_workspaces([workspace], ["R"], hasher=_hasher)

    assert out.exists()
    assert out.read_text() == "NEW"
    assert not root.exists()
    assert [path for _, path, _, _ in report.replaced] == [normpath(str(out))]


def test_missing_workspace_is_reported(tmp_path):
    root = tmp_path / "gone-ws"
    workspace = load_workspace("X", str(root))

    report = cleanup_workspaces([workspace], ["X"], hasher=_hasher)

    assert report.missing_workspaces == ["X"]
    assert not report.broken_references
    assert not report.removed_workspaces


def test_removing_all_workspaces_reclaims_shared_output(layout):
    ws_a, ws_b, ext, root_a, root_b = layout

    report = cleanup_workspaces([ws_a, ws_b], ["A", "B"], hasher=_hasher)

    assert not root_a.exists()
    assert not root_b.exists()
    assert not ext.exists()
    assert not report.protected


def test_protected_directory_inside_private_root_is_pruned(tmp_path):
    root_a = tmp_path / "wa"
    root_b = tmp_path / "wb"
    shared_dir = root_a / "data"
    kept = shared_dir / "kept"
    other = root_a / "other.txt"
    _write(str(kept), "keep")
    _write(str(other), "mine")
    # Survivor B references a directory that physically lives inside A's root.
    ws_a = _workspace("A", str(root_a), **{str(other): "mine"})
    ws_b = WorkspaceOutputs(
        "B",
        str(root_b),
        {normpath(str(shared_dir)): LockedOutput(normpath(str(shared_dir)), "abc.dir")},
    )

    report = cleanup_workspaces([ws_a, ws_b], ["A"], hasher=_hasher)

    assert kept.exists()
    assert not other.exists()
    assert root_a.is_dir()
    assert not report.removed_workspaces


def test_no_cache_outputs_are_protected_by_path(tmp_path):
    root_a = tmp_path / "wa"
    root_b = tmp_path / "wb"
    shared = tmp_path / "shared" / "metrics.yaml"
    _write(str(shared), "m")
    ws_a = WorkspaceOutputs(
        "A",
        str(root_a),
        {normpath(str(shared)): LockedOutput(normpath(str(shared)), None, False)},
    )
    ws_b = WorkspaceOutputs(
        "B",
        str(root_b),
        {normpath(str(shared)): LockedOutput(normpath(str(shared)), None, False)},
    )

    report = cleanup_workspaces([ws_a, ws_b], ["A"], hasher=_hasher)

    assert shared.exists()
    assert report.protected == [normpath(str(shared))]


def test_parse_lockfile_handles_wdir_and_no_cache(tmp_path):
    root = tmp_path / "w"
    root.mkdir()
    lockfile = root / "dvc.lock"
    lockfile.write_text(
        "stages:\n"
        "  train:\n"
        "    wdir: sub\n"
        "    outs:\n"
        "      - path: model.pkl\n"
        "        md5: abc123\n"
        "        size: 10\n"
        "      - path: metrics.txt\n"
        "        cache: false\n",
        encoding="utf-8",
    )

    outputs = parse_locked_outputs(str(lockfile), str(root))

    model = normpath(str(root / "sub" / "model.pkl"))
    metrics = normpath(str(root / "sub" / "metrics.txt"))
    assert outputs[model].checksum == "abc123"
    assert outputs[metrics].checksum is None
    assert outputs[metrics].use_cache is False


def test_load_workspace_marks_missing_root(tmp_path):
    workspace = load_workspace("Z", str(tmp_path / "missing"))
    assert workspace.exists is False
    assert workspace.outputs == {}
