"""Streaming clients for public-domain / openly-licensed melody APIs.

These clients only ever hold one HTTP response page in memory at a time and
yield raw ABC tune strings one at a time — nothing is written to disk here;
the caller (`pipeline.py`) tokenizes each tune immediately and discards the
text.

IMPORTANT — sandbox limitation: this development environment's outbound
network access is restricted to github.com/npm/pypi hosts, so the exact JSON
response shapes below were implemented from the documented/expected API
contracts and verified here only against mocked responses (see
`tests/test_data_sources.py`). Before relying on this in production, run it
once against the live APIs on a machine with normal internet access and
confirm the field names below still match (`abc_field_candidates`, and
thesession's `settings[].abc` shape) — see README.md "Verifying live data
sources" section.
"""
from __future__ import annotations

import logging
from typing import Iterator, Protocol

import requests

logger = logging.getLogger(__name__)


class MelodySource(Protocol):
    def stream_abc(self, params: dict, max_tunes: int) -> Iterator[str]:
        ...


def _looks_like_abc(text: str) -> bool:
    return isinstance(text, str) and ("K:" in text)


class TheSessionSource:
    """thesession.org — a CC0-licensed, community-maintained database of
    (mostly Irish/Celtic traditional) tunes in ABC notation, with a plain
    JSON REST API (append `format=json`)."""

    BASE_URL = "https://thesession.org/tunes/search"

    def __init__(self, session: requests.Session | None = None, timeout: float = 15.0):
        self._session = session or requests.Session()
        self._timeout = timeout

    def stream_abc(self, params: dict, max_tunes: int) -> Iterator[str]:
        page = 1
        yielded = 0
        per_page = min(50, max_tunes) or 1
        while yielded < max_tunes:
            query = {**params, "format": "json", "perpage": per_page, "page": page}
            resp = self._session.get(self.BASE_URL, params=query, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            tunes = data.get("tunes") or []
            if not tunes:
                break
            for tune in tunes:
                for setting in tune.get("settings") or []:
                    abc_text = setting.get("abc")
                    if not abc_text:
                        continue
                    yield self._as_full_abc(tune, setting, abc_text, params)
                    yielded += 1
                    if yielded >= max_tunes:
                        return
            total_pages = data.get("pages", page)
            if page >= total_pages:
                break
            page += 1

    @staticmethod
    def _as_full_abc(tune: dict, setting: dict, abc_body: str, params: dict) -> str:
        """thesession settings sometimes return the tune body alone (no
        header block); synthesize minimal headers from the surrounding tune/
        setting metadata when that happens so `abc_tokenizer.parse_abc` has
        a meter/key to work with."""
        if "K:" in abc_body:
            return abc_body
        name = tune.get("name", "Unknown")
        meter = tune.get("meter") or params.get("meter") or "4/4"
        key = setting.get("key") or "Cmaj"
        header = f"X:1\nT:{name}\nM:{meter}\nL:1/8\nK:{key}\n"
        return header + abc_body


class HuggingFaceDatasetSource:
    """Streams rows from a public Hugging Face dataset via the free
    datasets-server `/rows` REST endpoint (no auth, no local download of the
    dataset), looking for a column holding a standard ABC-notation string."""

    BASE_URL = "https://datasets-server.huggingface.co/rows"
    DEFAULT_ABC_FIELDS = ("abc", "output", "input", "text")

    def __init__(
        self,
        dataset: str,
        config: str = "default",
        split: str = "train",
        abc_fields: tuple[str, ...] | None = None,
        session: requests.Session | None = None,
        timeout: float = 20.0,
    ):
        self.dataset = dataset
        self.config = config
        self.split = split
        self.abc_fields = abc_fields or self.DEFAULT_ABC_FIELDS
        self._session = session or requests.Session()
        self._timeout = timeout

    def stream_abc(self, params: dict, max_tunes: int) -> Iterator[str]:
        offset = int(params.get("offset", 0))
        page_length = min(100, max_tunes) or 1
        require_4_4 = params.get("require_4_4", True)
        yielded = 0

        while yielded < max_tunes:
            query = {
                "dataset": self.dataset,
                "config": self.config,
                "split": self.split,
                "offset": offset,
                "length": page_length,
            }
            resp = self._session.get(self.BASE_URL, params=query, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("rows") or []
            if not rows:
                break

            for row_entry in rows:
                row = row_entry.get("row") or {}
                abc_text = self._extract_abc(row)
                if abc_text is None:
                    continue
                if require_4_4 and "M:4/4" not in abc_text:
                    continue
                yield abc_text
                yielded += 1
                if yielded >= max_tunes:
                    return

            offset += len(rows)
            total = data.get("num_rows_total")
            if total is not None and offset >= total:
                break

    def _extract_abc(self, row: dict) -> str | None:
        for field in self.abc_fields:
            value = row.get(field)
            if _looks_like_abc(value):
                return value
        return None


class FixtureSource:
    """In-memory source used for offline development/tests and for the
    `--source fixtures` CLI smoke-test mode: just replays a fixed list of
    ABC strings instead of calling the network."""

    def __init__(self, abc_tunes: list[str]):
        self._abc_tunes = abc_tunes

    def stream_abc(self, params: dict, max_tunes: int) -> Iterator[str]:
        for abc_text in self._abc_tunes[:max_tunes]:
            yield abc_text


def build_source(source_config: dict) -> MelodySource:
    """Factory used by `pipeline.py` to construct a source from
    `config/sources.yaml` entries."""
    kind = source_config["source"]
    if kind == "thesession":
        return TheSessionSource()
    if kind == "huggingface":
        return HuggingFaceDatasetSource(dataset=source_config["dataset"])
    if kind == "fixtures":
        return FixtureSource(source_config.get("fixtures", []))
    raise ValueError(f"Unknown data source kind: {kind!r}")
