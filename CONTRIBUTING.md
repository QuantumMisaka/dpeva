# Contributing to DP-EVA

Contributions in English or Chinese are welcome: documentation fixes, reproducible
bug reports, examples, tests, and workflow improvements all help.

## Choose an entry point

- Ask usage questions or discuss scientific workflows in
  [Discussions](https://github.com/QuantumMisaka/dpeva/discussions).
- Report a reproducible bug or propose a feature using the
  [Issue forms](https://github.com/QuantumMisaka/dpeva/issues/new/choose).
  Search existing issues first; discuss substantial architecture or public API
  changes before investing in an implementation.
- Small fixes can go directly to a pull request. Useful first contributions
  include correcting the quickstart, improving an error message, or adding a
  regression for a confirmed bug.

## Prepare your checkout

Fork the repository on GitHub, clone your fork, and create a topic branch:

```bash
git clone https://github.com/YOUR-USERNAME/dpeva.git
cd dpeva
git switch -c fix/short-description
```

Follow the [installation guide](docs/guides/installation.md) for the Python and
DeepMD backend environment, then install the development tools:

```bash
python -m pip install -e '.[dev]'
# For documentation changes:
python -m pip install -e '.[docs]'
```

A personal SAI account, GPU, or an environment named `dpeva-dpa4` is not required
for routine contributions. See the [developer guide](docs/guides/developer-guide.md)
for architecture and coding conventions, and [tests/README.md](tests/README.md)
for test layers.

## Make and verify a focused change

Add regression coverage at the layer that owns the behavior. Use temporary
paths and small fixtures; mock expensive training and cluster submissions.
Keep reusable examples in [examples/recipes](examples/recipes/README.md), with
paths relative to the configuration file. Avoid adding one-off smoke runners
or tests that merely assert report wording.

```bash
python scripts/run_gate.py local
python scripts/run_gate.py integration
# For documentation changes:
python scripts/run_gate.py docs_pr
```

Run focused tests while developing. The routine integration gate skips real
training and hardware checks unless explicitly enabled; see
[the integration guide](tests/integration/README.md). Do not enable these tests
on a shared login node. Record what was tested and any untested runtime boundary.

Changes to CLI behavior, configuration, outputs, or dependencies must update
relevant documentation and recipes in the same PR. Keep field definitions in
Reference and executable configurations in examples instead of duplicating them.

## Open and revise a pull request

Push your topic branch to your fork and open a PR targeting `main`. Fill in the
PR template: describe the problem, resulting behavior, tests, and migration
impact. Link the related issue when there is one. Draft PRs are welcome for
unfinished work or early feedback.

The normal merge path requires:

- One approving review; new commits dismiss stale approvals.
- Successful GitHub Actions checks: `lint`, `unit-tests`, `integration-tests`,
  `python-audit`, `explore-extra-smoke`, `build-docs`, and `doc-lint`.
- A branch up to date with `main` and resolved review conversations.

Required workflows run on every PR, including documentation-only changes.
Do not use CI-skip commit messages. A first-time fork contribution may need a
maintainer to approve the Actions run. Optional GPU/experimental checks are not
routine contribution prerequisites. Maintainers review changes and merge PRs;
respond to review with focused follow-up commits.

Administrators retain emergency bypass while the project has a single primary
maintainer. It is not the normal contribution path; any use should explain the
reason and verification in the PR. Force pushes and deletion of `main` are disabled.

## Licensing and attribution

The project uses [LGPL v3](LICENCE). Contribute only material you have the right
to share, and retain relevant third-party notices. For academic use, see
[CITATION.cff](CITATION.cff).
