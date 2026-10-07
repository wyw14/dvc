import json
import logging
from types import SimpleNamespace

import pytest

from dvc.exceptions import URLMissingError
from dvc.fs.paginate import (
    Entries,
    InvalidPageTokenError,
    Page,
    PaginatedListing,
    fs_list_page,
)
from dvc.repo.ls_url import ls_url


class ScriptedRemote:
    """A remote object store serving scripted pages keyed by request token."""

    def __init__(self, pages, *, fail=None):
        self.pages = pages
        self.fail = dict(fail or {})
        self.requests = []

    def fetch(self, path, token):
        self.requests.append(token)
        if token in self.fail:
            raise self.fail[token]
        return self.pages[token]

    def runner(self, path="/bucket"):
        return PaginatedListing(path, self.fetch)

    def run(self, path="/bucket"):
        return self.runner(path).run()


def file_info(name, size=None):
    info = {"name": name, "type": "file"}
    if size is not None:
        info["size"] = size
    return info


def dir_info(name):
    return {"name": name, "type": "directory", "size": 0}


def test_multi_page_traversal_advances_by_token():
    pages = {
        None: Page(files=[file_info("a")], next_token="t1"),
        "t1": Page(dirs=[dir_info("d/")], files=[file_info("b")], next_token="t2"),
        "t2": Page(files=[file_info("c")], next_token=None),
    }
    remote = ScriptedRemote(pages)

    listing = remote.run()

    assert listing.complete
    assert listing.pages == 3
    assert remote.requests == [None, "t1", "t2"]
    assert set(listing.files) == {"a", "b", "c"}
    assert set(listing.dirs) == {"d/"}
    assert [i["type"] for i in listing.entries()] == [
        "directory",
        "file",
        "file",
        "file",
    ]


def test_duplicate_names_recorded_once_within_and_across_pages():
    pages = {
        None: Page(
            dirs=[dir_info("d/")],
            files=[file_info("a"), file_info("a")],
            next_token="t1",
        ),
        "t1": Page(files=[file_info("a"), file_info("b")], next_token=None),
    }
    remote = ScriptedRemote(pages)

    listing = remote.run()

    assert listing.complete
    assert set(listing.files) == {"a", "b"}
    assert set(listing.dirs) == {"d/"}
    assert [i["name"] for i in listing.entries()].count("a") == 1


def test_same_page_is_never_requested_twice_on_non_advancing_token():
    pages = {
        # remote keeps pointing back at the token of the page it just returned
        None: Page(files=[file_info("a")], next_token="t1"),
        "t1": Page(files=[file_info("b")], next_token="t1"),
    }
    remote = ScriptedRemote(pages)

    runner = remote.runner()
    listing = runner.run()

    assert listing.incomplete
    assert isinstance(listing.error, InvalidPageTokenError)
    assert remote.requests == [None, "t1"]
    assert set(listing.files) == {"a", "b"}

    # resuming must not issue the same request again
    runner.run()
    assert remote.requests == [None, "t1"]
    assert set(listing.files) == {"a", "b"}


def test_empty_token_finishes_traversal_without_extra_requests():
    pages = {
        None: Page(dirs=[dir_info("d/")], files=[file_info("a")], next_token=""),
    }
    remote = ScriptedRemote(pages)

    listing = remote.run()

    assert listing.complete
    assert remote.requests == [None]
    assert listing.names() == {"d/", "a"}


def test_s3_style_response_is_partitioned_by_object_kind():
    response = {
        "CommonPrefixes": [{"Prefix": "dir/"}],
        "Contents": [{"Key": "obj"}],
        "NextContinuationToken": "",
    }
    page = Page.from_response(response)
    assert page.next_token is None
    assert {i["name"] for i in page.dirs} == {"dir/"}
    assert page.dirs[0]["type"] == "directory"
    assert {i["name"] for i in page.files} == {"obj"}
    assert page.files[0]["type"] == "file"


def test_tuple_response_shapes_supported():
    page = Page.from_response(
        (
            [dir_info("d/")],
            [file_info("a"), file_info("b")],
            "",
        )
    )
    assert {i["name"] for i in page.dirs} == {"d/"}
    assert {i["name"] for i in page.files} == {"a", "b"}
    assert page.next_token is None

    page = Page.from_response(
        (
            [file_info("a"), dir_info("d/"), file_info("b")],
            None,
        )
    )
    assert {i["name"] for i in page.dirs} == {"d/"}
    assert {i["name"] for i in page.files} == {"a", "b"}


def test_invalid_token_keeps_confirmed_pages_and_flags_incomplete(caplog):
    pages = {
        None: Page(files=[file_info("a")], next_token="bad"),
    }
    remote = ScriptedRemote(pages, fail={"bad": InvalidPageTokenError("no such token")})

    with caplog.at_level(logging.WARNING):
        listing = remote.run()

    assert listing.incomplete
    assert isinstance(listing.error, InvalidPageTokenError)
    assert set(listing.files) == {"a"}
    assert "incomplete" in caplog.text


def test_first_page_failure_surfaces_original_error():
    remote = ScriptedRemote({}, fail={None: FileNotFoundError("no such bucket")})

    with pytest.raises(FileNotFoundError):
        remote.run()


def test_dropped_connection_retains_pages_and_resumable_run():
    pages = {
        None: Page(files=[file_info("a")], next_token="t1"),
        "t1": Page(files=[file_info("b")], next_token=None),
    }
    remote = ScriptedRemote(
        {None: pages[None]}, fail={"t1": ConnectionError("reset by peer")}
    )
    runner = PaginatedListing("/bucket", remote.fetch)
    listing = runner.run()

    assert listing.incomplete
    assert listing.pages == 1
    assert set(listing.files) == {"a"}
    assert listing.token == "t1"

    # connection restored: resume from the failed token, confirmed entries stay
    remote.pages["t1"] = pages["t1"]
    del remote.fail["t1"]
    listing = runner.run()

    assert listing.complete
    assert set(listing.files) == {"a", "b"}
    assert remote.requests == [None, "t1", "t1"]


def test_repeated_failed_resume_never_loses_confirmed_pages():
    pages = {None: Page(files=[file_info("a")], next_token="t1")}
    remote = ScriptedRemote(pages, fail={"t1": TimeoutError()})
    runner = PaginatedListing("/bucket", remote.fetch)

    first = runner.run()
    second = runner.run()

    assert first.incomplete
    assert second.incomplete
    assert set(first.files) == set(second.files) == {"a"}


def test_empty_remote_is_stable_and_complete():
    remote = ScriptedRemote({None: Page(next_token=None)})
    listing = remote.run()

    assert listing.complete
    assert listing.pages == 1
    assert listing.entries() == []
    assert remote.requests == [None]


def test_single_object_remote_is_stable_across_runs():
    pages = {None: Page(files=[file_info("only")], next_token=None)}
    remote = ScriptedRemote(pages)
    runner = remote.runner()
    listing = runner.run()

    assert listing.complete
    assert set(listing.names()) == {"only"}

    listing2 = runner.run()
    assert listing2.complete
    assert listing2.entries() == listing.entries()
    # completed traversals are never fetched again
    assert remote.requests == [None]


def test_entries_behaves_as_plain_list_and_carries_incomplete_flag():
    entries = Entries([{"path": "a"}], incomplete=True)

    assert entries == [{"path": "a"}]
    assert entries.incomplete
    assert json.dumps(entries) == '[{"path": "a"}]'

    complete = Entries([{"path": "a"}])
    assert complete == [{"path": "a"}]
    assert not complete.incomplete


def test_fs_list_page_fallback_single_page():
    from dvc_objects.fs.memory import MemoryFileSystem

    fs = MemoryFileSystem()
    fs.pipe({"/data/a": b"a", "/data/dir/b": b"bb"})

    page = Page.from_response(fs_list_page(fs, "/data", None))
    assert page.next_token is None
    assert {i["name"] for i in page.files} == {"/data/a"}
    assert {i["name"] for i in page.dirs} == {"/data/dir"}

    with pytest.raises(InvalidPageTokenError):
        fs_list_page(fs, "/data", "unexpected-token")


class FakeRemoteFS:
    """Minimal fsspec-like fs served by scripted token-paginated pages."""

    protocol = "fakeremote"
    sep = "/"
    root_marker = "/"
    flavour = SimpleNamespace(sep="/")

    def __init__(self, pages, missing=False):
        # pages: {normalized dir path: list of page responses}
        self._pages = pages
        self._cursors = {}
        self.missing = missing
        self.calls = []

    @staticmethod
    def name(path):
        return path.rstrip("/").rsplit("/", 1)[-1]

    @classmethod
    def parts(cls, path):
        stripped = path.strip("/")
        return () if not stripped else tuple(stripped.split("/"))

    def relparts(self, path, start):
        p, s = self.parts(path), self.parts(start)
        return p[len(s) :]

    def relpath(self, path, start):
        parts = self.relparts(path, start)
        return "/".join(parts) if parts else "."

    def info(self, path):
        path = path.rstrip("/") or "/"
        if self.missing:
            raise FileNotFoundError(path)
        if path in self._pages:
            return {"name": path + "/", "type": "directory"}
        raise FileNotFoundError(path)

    def list_page(self, path, *, token, **kwargs):
        path = path.rstrip("/") or "/"
        self.calls.append((path, token))
        pages = self._pages[path]
        index = 0 if token is None else int(token)
        page = pages[index]
        self._cursors[path] = index + 1
        return page


def _pages(root_files, root_dirs=(), *, tokenized=False, extra=None):
    if not tokenized:
        return {
            "/data": [
                {
                    "dirs": [dir_info(f"/data/{d}/") for d in root_dirs],
                    "files": [file_info(f"/data/{f}") for f in root_files],
                    "next_token": None,
                }
            ],
            **(extra or {}),
        }
    first, second = root_files
    return {
        "/data": [
            {
                "dirs": [dir_info("/data/subdir/")],
                "files": [file_info(f"/data/{f}") for f in first],
                "next_token": "1",
            },
            {
                "dirs": [],
                "files": [file_info(f"/data/{f}") for f in second],
                "next_token": None,
            },
        ],
        **(extra or {}),
    }


@pytest.fixture
def patched_remote(mocker):
    def _patch(remote):
        mocker.patch(
            "dvc.repo.ls_url.parse_external_url",
            return_value=(remote, "/data"),
        )
        return remote

    return _patch


def test_ls_url_multi_page_non_recursive(patched_remote):
    remote = FakeRemoteFS(_pages((["a"], ["b"]), root_dirs=("subdir",), tokenized=True))
    patched_remote(remote)

    entries = ls_url("fake:///data")

    assert {(e["path"], e["isdir"]) for e in entries} == {
        ("a", False),
        ("b", False),
        ("subdir", True),
    }
    assert not entries.incomplete
    assert [c for _, c in remote.calls] == [None, "1"]


def test_ls_url_multi_page_recursive_and_maxdepth(patched_remote):
    sub_page = {
        "/data/subdir": [
            {
                "dirs": [],
                "files": [file_info("/data/subdir/c")],
                "next_token": None,
            }
        ]
    }
    remote = FakeRemoteFS(
        _pages((["a"], ["b"]), root_dirs=("subdir",), tokenized=True, extra=sub_page)
    )
    patched_remote(remote)

    recursive = ls_url("fake:///data", recursive=True)
    assert {(e["path"], e["isdir"]) for e in recursive} == {
        ("a", False),
        ("b", False),
        ("subdir/c", False),
    }
    assert not recursive.incomplete

    level1 = ls_url("fake:///data", recursive=True, maxdepth=1)
    assert {(e["path"], e["isdir"]) for e in level1} == {
        ("a", False),
        ("b", False),
        ("subdir", True),
    }

    level2 = ls_url("fake:///data", recursive=True, maxdepth=2)
    assert {(e["path"], e["isdir"]) for e in level2} == {
        ("a", False),
        ("b", False),
        ("subdir/c", False),
    }


def test_ls_url_duplicate_names_across_pages_deduped(patched_remote):
    pages = {
        "/data": [
            {"dirs": [], "files": [file_info("/data/a")], "next_token": "1"},
            {
                "dirs": [],
                "files": [file_info("/data/a"), file_info("/data/b")],
                "next_token": None,
            },
        ]
    }
    patched_remote(FakeRemoteFS(pages))

    entries = ls_url("fake:///data")

    assert [e["path"] for e in entries] == ["a", "b"]


def test_ls_url_invalid_token_returns_confirmed_entries_incomplete(
    patched_remote, caplog
):
    class InvalidTokenRemote(FakeRemoteFS):
        def list_page(self, path, *, token, **kwargs):
            self.calls.append((path, token))
            if token is None:
                return {
                    "dirs": [],
                    "files": [file_info("/data/a")],
                    "next_token": "bad",
                }
            raise InvalidPageTokenError("invalid token")

    patched_remote(InvalidTokenRemote({"/data": []}))

    with caplog.at_level(logging.WARNING):
        entries = ls_url("invalid:///data")

    assert [e["path"] for e in entries] == ["a"]
    assert entries.incomplete
    assert isinstance(entries.error, InvalidPageTokenError)
    assert "incomplete" in caplog.text


def test_ls_url_connection_drop_mid_level_retains_confirmed(patched_remote):
    class DroppingRemote(FakeRemoteFS):
        def __init__(self):
            super().__init__({"/data": []})

        def list_page(self, path, *, token, **kwargs):
            self.calls.append((path, token))
            path = path.rstrip("/")
            if path == "/data" and token is None:
                return {
                    "dirs": [dir_info("/data/lost/")],
                    "files": [file_info("/data/a")],
                    "next_token": None,
                }
            raise ConnectionError("reset by peer")

    remote = DroppingRemote()
    patched_remote(remote)

    entries = ls_url("drop:///data", recursive=True)

    # root level was confirmed; the lost subtree must not erase it
    assert {(e["path"], e["isdir"]) for e in entries} == {
        ("a", False),
    }
    assert entries.incomplete
    assert isinstance(entries.error, ConnectionError)


def test_ls_url_empty_remote_and_single_object_stable(patched_remote):
    patched_remote(FakeRemoteFS(_pages([])))
    assert ls_url("empty:///data") == []

    patched_remote(FakeRemoteFS(_pages(["only"])))
    entries = ls_url("single:///data")
    assert entries == [{"path": "only", "isdir": False, "size": None}]
    assert not entries.incomplete


def test_ls_url_missing_target_raises_original_error(patched_remote):
    patched_remote(FakeRemoteFS({}, missing=True))

    with pytest.raises(URLMissingError):
        ls_url("missing:///data")


def test_ls_url_repeated_runs_are_identical(patched_remote):
    sub_page = {
        "/data/subdir": [
            {
                "dirs": [],
                "files": [file_info("/data/subdir/c")],
                "next_token": None,
            }
        ]
    }
    remote = FakeRemoteFS(
        _pages((["a"], ["b"]), root_dirs=("subdir",), tokenized=True, extra=sub_page)
    )
    patched_remote(remote)

    first = ls_url("fake:///data", recursive=True)
    second = ls_url("fake:///data", recursive=True)

    assert list(first) == list(second)
    assert [(p, i) for p, i in remote.calls] == [
        ("/data", None),
        ("/data", "1"),
        ("/data/subdir", None),
        ("/data", None),
        ("/data", "1"),
        ("/data/subdir", None),
    ]


def test_ls_tree_marks_subtree_incomplete_but_keeps_siblings(monkeypatch):
    from dvc.repo.ls import _ls_tree
    from dvc_objects.fs.memory import MemoryFileSystem

    fs = MemoryFileSystem(global_store=False)
    fs.pipe({"/a": b"a", "/dir/x": b"x"})
    original_ls = fs.ls

    def flaky_ls(path, detail=True, **kwargs):
        if path.rstrip("/") == "/dir":
            raise FileNotFoundError(path)
        return original_ls(path, detail=detail, **kwargs)

    monkeypatch.setattr(fs, "ls", flaky_ls)

    tree = _ls_tree(fs, "/")
    root = tree["/"]
    contents = root["contents"]

    assert "a" in contents
    assert "contents" not in contents["a"]
    assert contents["dir"]["incomplete"] is True
    assert "contents" not in contents["dir"]
