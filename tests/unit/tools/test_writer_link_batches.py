"""Batched link writers must persist every input row with its data intact.

Each writer here turns a list of link rows into UNWIND batch queries. These
tests guarantee, per writer:
  - every input row lands in exactly one batch (no row lost or duplicated),
  - row values and defaults survive the conversion,
  - rows are routed to the query matching their labels,
  - a failing label combination does not abort the remaining ones.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from codegraphcontext.tools.indexing.persistence.writer import GraphWriter


class _FakeResult:
    def single(self):
        return None

    def __iter__(self):
        return iter([])


class _RecordingSession:
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
        return "falkordb"


def _make_writer(
    raise_for: Optional[str] = None, error: Optional[Exception] = None
) -> Tuple[GraphWriter, _RecordingSession]:
    session = _RecordingSession(raise_for=raise_for, error=error)
    return GraphWriter(driver=_FakeDriver(session), db_manager=_FakeDbManager()), session


def _batched_rows(session: _RecordingSession, token: str) -> List[Dict[str, Any]]:
    """All rows sent to queries whose text contains `token`."""
    rows = []
    for call in session.calls:
        if token in call["query"]:
            rows.extend(call["kwargs"].get("batch", []))
    return rows


# ---------------------------------------------------------------------------
# write_implements_links / write_partial_of_links — label-grouped writers
# ---------------------------------------------------------------------------

def test_implements_rows_are_grouped_by_label_pair():
    writer, session = _make_writer()
    writer.write_implements_links(
        [
            {
                "child_label": "Struct",
                "parent_label": "Interface",
                "child_name": "Dog",
                "path": "/repo/a.go",
                "parent_name": "Animal",
                "resolved_parent_file_path": "/repo/b.go",
            },
            {
                "child_label": "Class",
                "parent_label": "Interface",
                "child_name": "Cat",
                "path": "/repo/c.kt",
                "parent_name": "Animal",
                "resolved_parent_file_path": "/repo/b.go",
                "confidence_label": "EXTRACTED",
            },
        ]
    )
    impl_calls = [c for c in session.calls if "IMPLEMENTS" in c["query"]]
    assert len(impl_calls) == 2, "One query per (child_label, parent_label) group"

    struct_call = next(c for c in impl_calls if "child:Struct" in c["query"])
    assert struct_call["kwargs"]["batch"] == [
        {
            "child_name": "Dog",
            "path": "/repo/a.go",
            "parent_name": "Animal",
            "resolved_parent_file_path": "/repo/b.go",
            "confidence_label": "INFERRED",
        }
    ]
    class_call = next(c for c in impl_calls if "child:Class" in c["query"])
    assert class_call["kwargs"]["batch"][0]["confidence_label"] == "EXTRACTED"


def test_partial_of_rows_carry_defaults_and_all_rows_are_written():
    writer, session = _make_writer()
    rows = [
        {
            "child_name": f"Part{i}",
            "path": f"/repo/p{i}.cs",
            "parent_name": "Whole",
            "resolved_parent_file_path": "/repo/whole.cs",
        }
        for i in range(3)
    ]
    writer.write_partial_of_links(rows)
    written = _batched_rows(session, "PARTIAL_OF")
    assert [r["child_name"] for r in written] == ["Part0", "Part1", "Part2"]
    assert all(r["confidence_label"] == "INFERRED" for r in written)


# ---------------------------------------------------------------------------
# Single-shape writers
# ---------------------------------------------------------------------------

def test_part_of_writes_every_row_once():
    writer, session = _make_writer()
    writer.write_part_of_links(
        [
            {"child_path": "/repo/part1.f90", "parent_path": "/repo/lib.f90"},
            {"child_path": "/repo/part2.f90", "parent_path": "/repo/lib.f90"},
        ]
    )
    written = _batched_rows(session, "PART_OF")
    assert written == [
        {"child_path": "/repo/part1.f90", "parent_path": "/repo/lib.f90"},
        {"child_path": "/repo/part2.f90", "parent_path": "/repo/lib.f90"},
    ]


def test_decorated_by_applies_context_and_line_defaults():
    writer, session = _make_writer()
    writer.write_decorated_by_links(
        [
            {
                "decorated_name": "handler",
                "decorated_path": "/repo/a.py",
                "decorated_line": 10,
                "decorator_name": "cached",
                "decorator_path": "/repo/deco.py",
            }
        ]
    )
    (row,) = _batched_rows(session, "DECORATED_BY")
    assert row["decorated_context"] == ""
    assert row["line_number"] == 10, "line_number defaults to decorated_line"
    query = next(c["query"] for c in session.calls if "DECORATED_BY" in c["query"])
    assert 'row.decorated_context = "" OR decorated.context = row.decorated_context' in query


def test_metaclass_rows_carry_defaults():
    writer, session = _make_writer()
    writer.write_metaclass_links(
        [
            {
                "child_name": "Model",
                "path": "/repo/a.py",
                "parent_name": "Meta",
                "resolved_parent_file_path": "/repo/meta.py",
            }
        ]
    )
    (row,) = _batched_rows(session, "METACLASS")
    assert row["line_number"] == 0
    assert row["confidence_label"] == "EXTRACTED"


def test_companion_of_writes_row_with_both_endpoints():
    writer, session = _make_writer()
    writer.write_companion_of_links(
        [
            {
                "companion_name": "Companion",
                "companion_path": "/repo/a.kt",
                "companion_line": 5,
                "owner_name": "Widget",
                "owner_path": "/repo/a.kt",
                "owner_line": 2,
            }
        ]
    )
    (row,) = _batched_rows(session, "COMPANION_OF")
    assert row["companion_line"] == 5
    assert row["owner_line"] == 2


def test_embeds_writes_row_with_line_default():
    writer, session = _make_writer()
    writer.write_embeds_links(
        [
            {
                "child_name": "Extended",
                "path": "/repo/a.go",
                "parent_name": "Base",
                "resolved_parent_file_path": "/repo/a.go",
            }
        ]
    )
    (row,) = _batched_rows(session, "EMBEDS")
    assert row["child_name"] == "Extended"
    assert row["line_number"] == 0


# ---------------------------------------------------------------------------
# C# inheritance rows
# ---------------------------------------------------------------------------

def test_csharp_rows_classify_interfaces_and_secondary_bases_as_implements():
    file_data = {
        "lang": "c_sharp",
        "path": "/repo/User.cs",
        "interfaces": [{"name": "IUser", "bases": []}],
        "classes": [
            # First base is a class -> INHERITS; IUser is an interface -> IMPLEMENTS
            {"name": "User", "bases": ["BaseUser", "IUser"]},
        ],
        "structs": [],
        "records": [],
    }
    implements, inherits = GraphWriter._collect_csharp_inheritance_rows(file_data)
    assert [(r["child_name"], r["base_name"]) for r in inherits] == [("User", "BaseUser")]
    assert [(r["child_name"], r["base_name"]) for r in implements] == [("User", "IUser")]


def test_csharp_rows_empty_for_non_csharp_files():
    implements, inherits = GraphWriter._collect_csharp_inheritance_rows(
        {"lang": "go", "path": "/repo/a.go", "classes": [{"name": "X", "bases": ["Y"]}]}
    )
    assert implements == [] and inherits == []


def test_csharp_write_sends_rows_and_survives_binder_failure():
    writer, session = _make_writer(
        raise_for="child:`Mixin`",
        error=RuntimeError("Binder exception: cannot find a valid label"),
    )
    rows = [{"child_name": "User", "path": "/repo/User.cs", "base_name": "IUser"}]
    writer._write_csharp_inheritance_rows(session, rows, [])
    impl_calls = [c for c in session.calls if "IMPLEMENTS" in c["query"]]
    assert any("child:`Class`" in c["query"] for c in impl_calls)
    assert any("child:`Extension`" in c["query"] for c in impl_calls), (
        "Labels after the binder-failing one must still be attempted"
    )
    assert all(c["kwargs"]["batch"] == rows for c in impl_calls)


# ---------------------------------------------------------------------------
# write_scip_call_edges
# ---------------------------------------------------------------------------

def _scip_files_data() -> Dict[str, Any]:
    return {
        "/repo/a.go": {
            "function_calls_scip": [
                {
                    "caller_symbol": "pkg/GetUser().",
                    "caller_file": "/repo/a.go",
                    "caller_line": 12,
                    "callee_name": "Validate",
                    "callee_file": "/repo/b.go",
                    "ref_line": 14,
                }
            ],
            "module_level_calls_scip": [
                {
                    "caller_file": "/repo/a.go",
                    "callee_name": "init",
                    "callee_file": "/repo/c.go",
                    "ref_line": 3,
                }
            ],
        }
    }


def test_scip_edges_resolve_caller_name_and_batch_all_label_pairs():
    writer, session = _make_writer()
    writer.write_scip_call_edges(_scip_files_data(), lambda s: s.split("/")[-1].rstrip("()."))

    fn_calls = [c for c in session.calls if "caller:`" in c["query"] and "caller:File" not in c["query"]]
    assert len(fn_calls) == 100, "caller 10 labels x callee 10 labels"
    for c in fn_calls:
        assert c["kwargs"]["batch"] == [
            {
                "caller_name": "GetUser",
                "caller_file": "/repo/a.go",
                "caller_line": 12,
                "callee_name": "Validate",
                "callee_file": "/repo/b.go",
                "ref_line": 14,
            }
        ]

    module_calls = [c for c in session.calls if "caller:File" in c["query"]]
    assert len(module_calls) == 10, "callee 10 labels for module-level edges"
    assert all(c["kwargs"]["batch"][0]["callee_name"] == "init" for c in module_calls)


def test_scip_edges_failure_on_one_label_pair_does_not_abort_the_rest():
    writer, session = _make_writer(
        raise_for="caller:`Variable`",
        error=RuntimeError("boom"),
    )
    writer.write_scip_call_edges(_scip_files_data(), lambda s: s)
    queries = [c["query"] for c in session.calls]
    assert any("caller:`Variable`" in q for q in queries)
    assert any("caller:`Extension`" in q for q in queries), (
        "Label pairs after a failing one must still be attempted"
    )


def test_scip_edges_no_queries_when_no_edges():
    writer, session = _make_writer()
    writer.write_scip_call_edges(
        {"/repo/a.go": {"function_calls_scip": [], "module_level_calls_scip": []}},
        lambda s: s,
    )
    assert session.calls == []
