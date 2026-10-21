# DockStream oracle — environment isolation note

The `DockStream` oracle (`oracles/docking/dockstream.py`) runs docking in a
**separate conda environment** (`DockStream`) as a subprocess, launched from the
`saturn` environment.

## OpenBabel variable leakage (BABEL_LIBDIR / BABEL_DATADIR)

Both the `saturn` environment and the `DockStream` environment ship their own
OpenBabel installation. When OpenBabel is installed via conda, activating an
environment can export `BABEL_LIBDIR` and `BABEL_DATADIR`. If these are
inherited by the DockStream subprocess, DockStream's OpenBabel loads **Saturn's**
(incompatible) plugins, which silently breaks ligand conversion and docking —
every molecule "fails" and is scored `0.0`.

**Fix (implemented in `dockstream.py`):** the subprocess is launched with an
explicitly sanitized environment that removes `BABEL_LIBDIR` and `BABEL_DATADIR`
(see `DockStream._build_subprocess_env`). Nothing else is changed, so docking
scores are identical to a correctly isolated manual setup.

## Failing loudly on systematic docking failures

Previously, any molecule whose docking failed was assigned a score of `0.0` and
the run continued. A *systematic* failure (such as the OpenBabel leakage above)
could therefore go unnoticed while the optimization chased meaningless scores.

The oracle now tracks the per-batch failure rate (see
`DockStream._check_failure_rate`):

* **Any** failures emit a `RuntimeWarning` reporting how many molecules failed.
* If **≥ 10%** of a batch fails docking, a `RuntimeError` is raised and the run
  terminates, with a message pointing at the most common cause (the OpenBabel
  variables above). The threshold is the module-level constant
  `_MAX_DOCKING_FAILURE_FRACTION` in `dockstream.py`.

A small number of failures (e.g. an occasional un-embeddable ligand) is normal
and only warns; a broken setup now fails fast instead of silently degrading the
results.
