# Import PyTorch (if installed) before scikit-learn/scipy. On macOS, a pip-installed
# torch and conda-installed scientific packages each ship their own OpenMP runtime,
# and loading the conda one first can make torch crash on its first CPU kernel.
try:
    import torch  # noqa: F401
except ImportError:
    pass
