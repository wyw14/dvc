"""Token-paginated traversal of remote object-storage listings.

Remote object stores (S3, GCS, OSS, ...) return listings one page at a time,
each page carrying a continuation/next-page token. A single page may contain
directory markers (e.g. S3 ``CommonPrefixes``) alongside regular objects, may
repeat an object name that was already seen on a previous page, and may report
an empty (or otherwise non-advancing) next-page token.

This module provides traversal primitives with the following guarantees:

* traversal advances strictly by the token returned by the remote, the first
  page is always requested without a token and the same request is never
  issued twice (a token that does not advance the listing stops traversal);
* directory markers and file objects are recorded in separate partitions and
  duplicate object names within or across pages are recorded exactly once;
* when the remote rejects a token or the connection drops after one or more
  pages have been confirmed, the confirmed entries are retained and the
  listing is flagged as incomplete instead of being presented as a full
  listing; the traversal can be resumed, previously confirmed entries are
  never discarded;
* empty remotes and remotes holding a single object produce stable results.

Filesystems may opt into token pagination by implementing
``list_page(path, *, token, **kwargs)`` returning a :class:`Page`-like
object. Filesystems without it are traversed as a single page per directory
level, which matches the behaviour of a regular ``fsspec`` ``ls`` call.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dvc.exceptions import DvcException
from dvc.log import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dvc_objects.fs.base import FileSystem

logger = logger.getChild(__name__)

#: Token value used when requesting the first page of a listing.
START_TOKEN: str | None = None

_DIR_KEYS = (
    "dirs",
    "directories",
    "common_prefixes",
    "CommonPrefixes",
    "prefixes",
)
_FILE_KEYS = (
    "files",
    "objects",
    "Objects",
    "contents",
    "Contents",
)
_TOKEN_KEYS = (
    "next_token",
    "next_continuation_token",
    "NextContinuationToken",
    "next_marker",
    "NextMarker",
    "nextPageToken",
)
_NAME_KEYS = ("name", "path", "Key", "Prefix")

PageEntry = dict[str, Any]
FetchPage = Callable[[str, str | None], Any]
KeyFn = Callable[[PageEntry], str]
WalkLevel = tuple[str, dict[str, PageEntry], dict[str, PageEntry]]


class InvalidPageTokenError(DvcException):
    """Raised when a remote rejects a pagination token or fails to advance."""


@dataclass
class Page:
    """A single confirmed page of a remote listing.

    ``dirs`` holds directory markers/prefixes, ``files`` holds regular
    objects and ``next_token`` is the token to use for the next page or
    ``None``/empty when there are no more pages.
    """

    dirs: list[PageEntry] = field(default_factory=list)
    files: list[PageEntry] = field(default_factory=list)
    next_token: str | None = None

    @staticmethod
    def _as_info(entry: Any, *, force_dir: bool = False) -> PageEntry:
        if isinstance(entry, str):
            is_dir = force_dir or entry.endswith("/")
            return {"name": entry, "type": "directory" if is_dir else "file"}

        info = dict(entry)
        name = next((info[key] for key in _NAME_KEYS if info.get(key)), None)
        if name is None:
            raise TypeError(f"listing entry is missing a name: {entry!r}")
        info["name"] = name

        mtype = info.get("type")
        if mtype == "dir":
            info["type"] = "directory"
        elif mtype not in ("directory", "file"):
            is_dir = force_dir or (isinstance(name, str) and name.endswith("/"))
            info["type"] = "directory" if is_dir else "file"
        return info

    @classmethod
    def from_response(cls, response: Any) -> Page:
        """Normalize a page response coming from a remote/filesystem.

        Accepted shapes are :class:`Page`, a 3-tuple
        ``(dirs, files, next_token)``, a 2-tuple ``(entries, next_token)``
        where entries are partitioned by their ``type`` or a mapping with
        any of the key aliases defined in this module.
        """
        if isinstance(response, Page):
            return cls(
                dirs=[dict(i) for i in response.dirs],
                files=[dict(i) for i in response.files],
                next_token=response.next_token or None,
            )

        dirs_raw: list[Any]
        files_raw: list[Any]
        token: Any
        if isinstance(response, dict):
            dirs_raw = next((response[k] for k in _DIR_KEYS if k in response), [])
            files_raw = next((response[k] for k in _FILE_KEYS if k in response), [])
            token = next((response[k] for k in _TOKEN_KEYS if response.get(k)), None)
        elif isinstance(response, (tuple, list)):
            if len(response) == 3:
                dirs_raw, files_raw, token = response
            elif len(response) == 2:
                entries_raw, token = response
                normalized = [cls._as_info(e) for e in entries_raw]
                dirs_raw = [e for e in normalized if e["type"] == "directory"]
                files_raw = [e for e in normalized if e["type"] != "directory"]
            else:
                raise TypeError(f"invalid listing page: {response!r}")
        else:
            raise TypeError(f"invalid listing page: {response!r}")

        return cls(
            dirs=[cls._as_info(e, force_dir=True) for e in dirs_raw],
            files=[cls._as_info(e) for e in files_raw],
            next_token=token or None,
        )


@dataclass
class Listing:
    """Accumulated, possibly incomplete result of a paginated traversal."""

    path: str
    dirs: dict[str, PageEntry] = field(default_factory=dict)
    files: dict[str, PageEntry] = field(default_factory=dict)
    pages: int = 0
    complete: bool = False
    error: BaseException | None = None
    token: str | None = START_TOKEN

    @property
    def incomplete(self) -> bool:
        return not self.complete

    def entries(self) -> list[PageEntry]:
        """All confirmed entries (directory markers followed by files)."""
        return list(self.dirs.values()) + list(self.files.values())

    def names(self) -> set[str]:
        return set(self.dirs) | set(self.files)

    def merge(self, page: Page, key: KeyFn) -> tuple[int, int]:
        """Merge one confirmed page, de-duplicating object names.

        Returns the number of new directory and file entries added.
        """
        new_dirs = 0
        new_files = 0
        for info in page.dirs:
            name = key(info)
            if name not in self.dirs:
                self.dirs[name] = info
                new_dirs += 1
        for info in page.files:
            name = key(info)
            if name not in self.files:
                self.files[name] = info
                new_files += 1
        return new_dirs, new_files


def _default_key(info: PageEntry) -> str:
    return info["name"]


class PaginatedListing:
    """Traverse a token-paginated listing while retaining confirmed pages.

    The same instance can be driven to completion with repeated
    :meth:`run` calls: a completed traversal is never re-fetched and entries
    confirmed before a failure are always retained.
    """

    def __init__(
        self,
        path: str,
        fetch_page: FetchPage,
        *,
        key: KeyFn | None = None,
    ) -> None:
        self.path = path
        self.fetch_page = fetch_page
        self.key = key or _default_key
        self.listing = Listing(path=path)
        self._used_tokens: set[str | None] = set()
        self._stale_tokens: set[str] = set()

    def _fail(self, exc: BaseException, token: str | None) -> None:
        """Record a mid-traversal failure.

        Raises again when no page has been confirmed yet (so that errors for
        missing paths surface in their original form); otherwise retains the
        confirmed pages and marks the listing incomplete.
        """
        listing = self.listing
        listing.complete = False
        listing.token = token
        listing.error = exc
        if listing.pages == 0:
            raise exc
        logger.warning(
            "Listing of '%s' is incomplete (%s); %d confirmed page(s) with "
            "%d entr%s retained, the remote may contain more",
            self.path,
            exc,
            listing.pages,
            len(listing.dirs) + len(listing.files),
            "y" if len(listing.dirs) + len(listing.files) == 1 else "ies",
        )

    def run(self) -> Listing:
        listing = self.listing
        if listing.complete:
            return listing

        while True:
            token = listing.token
            if token is not None and token in self._stale_tokens:
                # The remote already proved this token does not advance the
                # listing; do not issue the same request again.
                self._fail(
                    InvalidPageTokenError(
                        f"listing token {token!r} does not advance the listing"
                    ),
                    token,
                )
                break

            try:
                response = self.fetch_page(self.path, token)
            except InvalidPageTokenError as exc:
                # The remote explicitly rejected the token. Keep the resume
                # token so a subsequent run() can retry the request.
                self._fail(exc, token)
                break
            except (ConnectionError, TimeoutError, OSError) as exc:
                # Connection dropped mid-traversal. Keep the resume token.
                self._fail(exc, token)
                break

            page = Page.from_response(response)
            listing.pages += 1
            listing.merge(page, self.key)

            next_token = page.next_token or None
            self._used_tokens.add(token)
            if next_token is None:
                listing.complete = True
                listing.token = None
                listing.error = None
                break

            if next_token == token or next_token in self._used_tokens:
                # Following the token would re-issue a request for a page that
                # was already fetched; stop without looping.
                self._stale_tokens.add(next_token)
                self._fail(
                    InvalidPageTokenError(
                        f"listing token {next_token!r} does not advance the listing"
                    ),
                    next_token,
                )
                break

            listing.token = next_token

        return listing


def fs_list_page(
    fs: FileSystem,
    path: str,
    token: str | None = None,
    **kwargs: Any,
) -> Any:
    """Fetch one listing page from an fsspec-like filesystem.

    Filesystems implementing ``list_page(path, *, token, **kwargs)`` are
    traversed with token pagination. Other filesystems return their full
    ``ls`` result as a single (final) page when the first page is requested
    and reject any further token.
    """
    list_page = getattr(fs, "list_page", None)
    if callable(list_page):
        return list_page(path, token=token, **kwargs)

    if token is None:
        infos = fs.ls(path, detail=True, **kwargs)
        dirs = [info for info in infos if info.get("type") == "directory"]
        files = [info for info in infos if info.get("type") != "directory"]
        return Page(dirs=dirs, files=files, next_token=None)

    raise InvalidPageTokenError(
        f"{fs.protocol if hasattr(fs, 'protocol') else fs!r} filesystem does "
        f"not support paginated listings but got token {token!r}"
    )


class Entries(list):
    """Listing entries that may come from an incomplete traversal.

    Behaves as a plain list (including JSON serialization and equality),
    while :attr:`incomplete`/:attr:`error` let callers tell confirmed
    complete listings apart from partial ones.
    """

    def __init__(
        self,
        iterable: Any = (),
        *,
        incomplete: bool = False,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(iterable)
        self.incomplete = incomplete
        self.error = error


class PaginatedWalk:
    """Breadth-first walk over a paginated filesystem.

    Yields ``(root, dirs, files)`` like ``fsspec`` ``walk`` with
    ``detail=True``, where ``dirs``/``files`` are ``{name: info}`` mappings.
    Each directory level is traversed page-by-page via continuation tokens.

    Entries from confirmed levels are always yielded. If a level cannot be
    fully listed (invalid token, dropped connection), :attr:`complete` is
    ``False``, :attr:`error` holds the failure and iteration stops after the
    already confirmed levels.
    """

    def __init__(
        self,
        fs: FileSystem,
        path: str,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.fs = fs
        self.path = path
        self.maxdepth = maxdepth
        self.ls_kwargs = kwargs
        self.complete = True
        self.error: BaseException | None = None

    def __iter__(self) -> Iterator[WalkLevel]:
        todo: deque[tuple[str, int]] = deque([(self.path, 0)])
        first_level = True

        def fetch(path: str, token: str | None) -> Any:
            return fs_list_page(self.fs, path, token, **self.ls_kwargs)

        while todo:
            root, depth = todo.popleft()
            runner = PaginatedListing(root, fetch)
            try:
                listing = runner.run()
            except (InvalidPageTokenError, OSError) as exc:
                if first_level:
                    raise
                self.complete = False
                self.error = exc
                logger.warning(
                    "Walk of '%s' is incomplete (%s); confirmed entries are "
                    "retained, the remote may contain more",
                    self.path,
                    exc,
                )
                return

            first_level = False
            if not listing.complete:
                # flag before yielding: non-recursive consumers stop after the
                # first level and must still see the listing as incomplete
                self.complete = False
                self.error = listing.error
            yield root, dict(listing.dirs), dict(listing.files)

            if not listing.complete:
                return

            if self.maxdepth is not None and depth + 1 >= self.maxdepth:
                continue

            for info in listing.dirs.values():
                todo.append((info["name"], depth + 1))
