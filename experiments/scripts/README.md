# Notebook-derived RAMMs checks

These scripts preserve the executable demonstrations that were previously embedded throughout `RAMMs-v3.ipynb`.

They are **manual research checks**, examples, and visual smoke tests—not a replacement for automated unit or integration tests. They belong well under `scripts/checks/` (or `examples/`) because many of them:

- print diagnostic information,
- open plots for human inspection,
- exercise exploratory configurations,
- do not yet have stable expected numerical answers.

Run them from this directory after installing RAMMs in editable mode:

```bash
pip install -e .
cd scripts/checks
python check_core.py
python check_contact.py
python check_symbolic.py
python check_workspace.py
python check_mobility.py
```

## Suggested repository placement

```text
ramms/
├── src/ramms/
├── scripts/
│   └── checks/
│       ├── _fixtures.py
│       ├── check_core.py
│       ├── check_plotting.py
│       ├── check_kinematics.py
│       ├── check_contact.py
│       ├── check_symbolic.py
│       ├── check_workspace.py
│       └── check_mobility.py
└── tests/
    ├── unit/
    └── integration/
```

As expected results become stable, move the corresponding logic into `tests/` and replace visual/printed inspection with assertions.

## Notes carried over from the notebook

- The node-segment examples were copied closely, including repeated segment choices that may have been notebook copy/paste mistakes. Review the four top-node cases in `check_contact.py` before treating them as canonical.
- The cascading-motion script calls `_get_fractional_centerline_position`, which is a private method. That is acceptable for a temporary diagnostic script but should not become part of the documented public API unless you rename it.
- These scripts assume the module names `core`, `contact`, `symbolic`, `workspace`, `mobility`, and `plotting` discussed during the package migration. Adjust imports if your final filenames differ.
