"""The app as it runs on Streamlit Community Cloud: started from an empty data folder,
it builds the synthetic demo database itself, and every chart and option then runs."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_builds_the_demo_database_and_every_option_runs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("YIELDDASH_DATA", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert (tmp_path / "demo.db").exists()
    source = next(s for s in at.sidebar.selectbox if s.label == "Data source")
    assert source.options == ["Synthetic demo"]
    assert any("Synthetic demo data, not real fab data" in box.value for box in at.info)
    assert len(at.tabs) >= 4

    checked = 0
    for i in range(len(at.radio)):
        for option in at.radio[i].options:
            at.radio[i].set_value(option).run()
            assert not at.exception, (at.radio[i].label, option, at.exception)
            checked += 1
    assert checked >= 3  # at least the three control chart types


def test_build_demo_replaces_an_existing_file(tmp_path):
    from yielddash.etl import build_demo

    db = tmp_path / "sub" / "demo.db"
    first = build_demo(db)
    second = build_demo(db)
    assert first == second and first["products"] == 3
    assert [p.name for p in db.parent.iterdir()] == ["demo.db"]  # no temporary file left behind


def test_requirements_match_pyproject():
    """Community Cloud installs requirements.txt, so it must list the same packages as pyproject.toml."""
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    listed = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert sorted(listed) == sorted(declared)
