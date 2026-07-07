"""Binder errors (missing label table on KuzuDB) must not abort a file write.

KuzuDB creates node tables lazily, so MATCH against a label with no table
raises a binder exception. The writer must skip that label combination and
keep writing the rest; any other error must propagate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from codegraphcontext.tools.indexing.persistence.writer import GraphWriter


class _FakeResult:
    def single(self):
        return None

    def __iter__(self):
        return iter([])


class _RecordingSession:
    """Records queries; raises `error` when `raise_for` is in the query text."""

    def __init__(self, raise_for: Optional[str] = None, error: Optional[Exception] = None):
        self.calls: List[Dict[str, Any]] = []
        self._raise_for = raise_for
        self._error = error

    def run(self, query: str, **kwargs):
        self.calls.append({"query": query, "kwargs": kwargs})
        if self._raise_for and self._raise_for in query:
            raise self._error
        return _FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _FakeDriver:
    def __init__(self, session: _RecordingSession):
        self._session = session

    def session(self):
        return self._session


class _FakeDbManager:
    def get_backend_type(self) -> str:
        return "kuzudb"


def _make_writer(
    raise_for: Optional[str] = None, error: Optional[Exception] = None
) -> Tuple[GraphWriter, _RecordingSession]:
    session = _RecordingSession(raise_for=raise_for, error=error)
    return GraphWriter(driver=_FakeDriver(session), db_manager=_FakeDbManager()), session


def _file_data_with_class_method() -> Dict[str, Any]:
    return {
        "path": "/repo/pkg/mod.py",
        "repo_path": "/repo",
        "lang": "python",
        "is_dependency": False,
        "functions": [
            {
                "name": "run",
                "line_number": 3,
                "args": [],
                "class_context": "Worker",
                "class_context_line": 2,
            }
        ],
        "classes": [{"name": "Worker", "line_number": 2}],
        "variables": [],
        "imports": [],
        "function_calls": [],
    }


def test_binder_error_on_one_label_does_not_abort_the_write():
    writer, session = _make_writer(
        raise_for="(c:Object",
        error=RuntimeError("Binder exception: cannot find a valid label"),
    )
    writer.add_file_to_graph(
        _file_data_with_class_method(), "repo", {}, repo_path_str="/repo"
    )
    class_fn_queries = [
        c["query"] for c in session.calls if "MERGE (c)-[:CONTAINS]->(fn)" in c["query"]
    ]
    assert any("(c:Object" in q for q in class_fn_queries)
    assert any("(c:Mixin" in q for q in class_fn_queries), (
        "Writes for the remaining labels must continue after a binder error"
    )


def test_non_binder_error_propagates():
    writer, _ = _make_writer(raise_for="(c:Class", error=RuntimeError("disk full"))
    with pytest.raises(RuntimeError, match="disk full"):
        writer.add_file_to_graph(
            _file_data_with_class_method(), "repo", {}, repo_path_str="/repo"
        )
