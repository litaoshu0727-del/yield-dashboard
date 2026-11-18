"""Smoke test: the Streamlit app runs end to end on the synthetic demo database.

Build the database first (CI does this before running the tests):

    python -m yielddash.etl --synthetic --db data/demo.db
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo.db"

pytestmark = pytest.mark.skipif(not DEMO.exists(), reason="no data/demo.db; build it with yielddash.etl --synthetic")


def test_every_chart_and_option_runs_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    source = next(s for s in at.sidebar.selectbox if s.label == "Data source")
    source.set_value("Synthetic demo").run()
    assert not at.exception, at.exception
    assert len(at.tabs) >= 4

    checked = 0
    for i in range(len(at.radio)):
        for option in at.radio[i].options:
            at.radio[i].set_value(option).run()
            assert not at.exception, (at.radio[i].label, option, at.exception)
            checked += 1
    assert checked >= 3  # at least the three control chart types
