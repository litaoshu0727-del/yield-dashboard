"""Read the WM-811K file (LSWMD.pkl), a Python 2 / pandas 0.x pickle current pandas cannot load."""
from __future__ import annotations

import copyreg
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class _Raw:
    """Stand-in for an old pandas class: keeps the pickled state instead of rebuilding it."""

    def __setstate__(self, state):
        self.state = state


def _raw_index(cls, fields):
    return fields


class _LegacyUnpickler(pickle.Unpickler):
    """Only numpy and a few builtins may be constructed; anything else in the file is refused."""

    _BUILTINS = {
        ("copy_reg", "_reconstructor"): copyreg._reconstructor,
        ("__builtin__", "object"): object,
        ("__builtin__", "slice"): slice,
    }

    def find_class(self, module, name):
        if module.startswith("pandas."):
            return _raw_index if name == "_new_Index" else _Raw
        if module == "numpy" or module.startswith("numpy."):
            return super().find_class(module.replace("numpy.core", "numpy._core"), name)
        if (module, name) in self._BUILTINS:
            return self._BUILTINS[(module, name)]
        raise pickle.UnpicklingError(f"unexpected object in pickle: {module}.{name}")


def _label(x) -> str | None:
    """Labels are stored as nested arrays: [['Center']], or [] when unlabeled."""
    arr = np.asarray(x).ravel()
    return str(arr[0]) if arr.size else None


def load_wm811k(path: str | Path) -> pd.DataFrame:
    """Return one row per wafer: waferMap (0 no die / 1 pass / 2 fail), lotName, waferIndex, failureType."""
    with open(path, "rb") as f:
        frame = _LegacyUnpickler(f, encoding="latin1").load()
    manager = frame.state["_data"].state[3]["0.14.1"]
    columns = [str(c) for c in manager["axes"][0]["data"]]
    data = {}
    for block in manager["blocks"]:
        positions = np.atleast_1d(np.arange(len(columns))[block["mgr_locs"]])
        for row, pos in zip(block["values"], positions):
            data[columns[pos]] = list(row) if row.dtype == object else row
    return pd.DataFrame({
        "waferMap": data["waferMap"],
        "lotName": [str(x) for x in data["lotName"]],
        "waferIndex": np.asarray(data["waferIndex"]).astype(int),
        "failureType": [_label(x) for x in data["failureType"]],
    })
