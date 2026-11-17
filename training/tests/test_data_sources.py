from unittest.mock import MagicMock

from training.src.data_sources import FixtureSource, HuggingFaceDatasetSource, TheSessionSource


def _fake_response(json_body):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=json_body)
    return resp


def test_fixture_source_replays_fixed_list():
    source = FixtureSource(["ABC_ONE", "ABC_TWO", "ABC_THREE"])
    assert list(source.stream_abc({}, max_tunes=2)) == ["ABC_ONE", "ABC_TWO"]
    assert list(source.stream_abc({}, max_tunes=10)) == ["ABC_ONE", "ABC_TWO", "ABC_THREE"]


def test_thesession_source_extracts_abc_and_paginates():
    page1 = {
        "tunes": [
            {
                "name": "The First Tune",
                "meter": "4/4",
                "settings": [{"key": "Dmaj", "abc": "X:1\nT:The First Tune\nM:4/4\nL:1/8\nK:D\nDEFG ABcd|"}],
            },
            {
                "name": "The Second Tune",
                "meter": "4/4",
                "settings": [{"key": "Ador", "abc": "ABcd efga|"}],  # no K: header -> must be synthesized
            },
        ],
        "pages": 2,
        "page": 1,
    }
    page2 = {
        "tunes": [
            {
                "name": "The Third Tune",
                "meter": "4/4",
                "settings": [{"key": "Gmaj", "abc": "GABc defg|"}],
            }
        ],
        "pages": 2,
        "page": 2,
    }

    session = MagicMock()
    session.get.side_effect = [_fake_response(page1), _fake_response(page2)]

    source = TheSessionSource(session=session)
    results = list(source.stream_abc({"type": "reel"}, max_tunes=10))

    assert len(results) == 3
    # First tune already had a K: header -> passed through unchanged.
    assert results[0].startswith("X:1\nT:The First Tune")
    # Second tune had no header -> synthesized with the right key/meter.
    assert "K:Ador" in results[1]
    assert "M:4/4" in results[1]
    assert "ABcd efga|" in results[1]
    # Pagination stopped after page 2 (data["pages"] == 2).
    assert session.get.call_count == 2


def test_thesession_source_respects_max_tunes_across_pagination():
    page1 = {
        "tunes": [
            {"name": "A", "meter": "4/4", "settings": [{"key": "C", "abc": "X:1\nK:C\nCDEF|"}]},
            {"name": "B", "meter": "4/4", "settings": [{"key": "C", "abc": "X:1\nK:C\nGABc|"}]},
        ],
        "pages": 5,
        "page": 1,
    }
    session = MagicMock()
    session.get.side_effect = [_fake_response(page1)]

    source = TheSessionSource(session=session)
    results = list(source.stream_abc({}, max_tunes=1))
    assert len(results) == 1
    assert session.get.call_count == 1


def test_huggingface_source_filters_by_meter_and_known_fields():
    page1 = {
        "rows": [
            {"row": {"output": "X:1\nM:4/4\nL:1/8\nK:C\nCDEF GABc|"}},
            {"row": {"output": "X:1\nM:3/4\nL:1/8\nK:C\nCDE FGA|"}},  # wrong meter, filtered out
            {"row": {"input": "X:1\nM:4/4\nL:1/8\nK:G\nGABc defg|"}},
            {"row": {"unrelated_field": "not abc at all"}},
        ],
        "num_rows_total": 4,
    }
    session = MagicMock()
    session.get.side_effect = [_fake_response(page1)]

    source = HuggingFaceDatasetSource(dataset="sander-wood/melodyhub", session=session)
    results = list(source.stream_abc({"require_4_4": True}, max_tunes=10))

    assert len(results) == 2
    assert all("M:4/4" in r for r in results)


def test_huggingface_source_paginates_with_offset():
    page1 = {"rows": [{"row": {"abc": f"X:1\nM:4/4\nK:C\nnote{i}|"}} for i in range(2)], "num_rows_total": 4}
    page2 = {"rows": [{"row": {"abc": f"X:1\nM:4/4\nK:C\nnote{i}|"}} for i in range(2, 4)], "num_rows_total": 4}
    session = MagicMock()
    session.get.side_effect = [_fake_response(page1), _fake_response(page2)]

    source = HuggingFaceDatasetSource(dataset="x/y", session=session)
    results = list(source.stream_abc({"require_4_4": True}, max_tunes=4))
    assert len(results) == 4
    assert session.get.call_count == 2
