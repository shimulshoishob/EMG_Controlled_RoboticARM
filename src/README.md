<div align="center">

# 📦 `src/`

**Source code of the desktop application**

[↑ Project](../README.md) · [Architecture](../docs/ARCHITECTURE.md)

</div>

The only package is [`emg_arm/`](emg_arm/README.md). It is kept under `src/` so imports always read `from emg_arm… import …` and tests can put this folder on `sys.path`.

```
src/
└── emg_arm/      the application: armband link, arm control, gesture logic, UI
```

Start the app from the project root with `python3 run.py`; `run.py` adds `src/` to the path.
