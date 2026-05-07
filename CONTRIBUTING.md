# Contributing to Seeds of Truth

Thank you for considering a contribution. This document explains how to set up a development environment, what kinds of changes we are looking for, and the process for getting code merged.

## Code of conduct

This project follows the [Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to abide by it. Report unacceptable behavior privately to the maintainers (see [SECURITY.md](SECURITY.md) for the contact address — the same channel handles conduct issues).

## What contributions are welcome

- **Bug reports** with a minimal reproduction.
- **Pull requests** that close an open issue, especially items in [ALPHA_BLOCKERS.md](ALPHA_BLOCKERS.md) and the v1.0 roadmap in the README.
- **Documentation improvements** — README clarifications, schema docs, deployment notes.
- **Retrieval-quality experiments** — new ranking heuristics, embeddings layer, alternative gating strategies. Use the eval harness to demonstrate the change is an improvement.
- **New LLM adapters** — implement `model_adapters.LLMStrategy` for additional backends.

If you are planning a change that touches more than ~100 lines or changes a public behavior, please open an issue first to discuss. We would rather sync on direction up front than ask you to rewrite a large PR.

## Development setup

### Prerequisites

- Python 3.11 (see `.python-version`)
- A C compiler (for spaCy model wheels)
- An LLM backend API key for at least one of: HuggingFace, DeepInfra. For most retrieval-only development, the `sim` adapter is sufficient and requires no credentials.
- A corpus database (a small sample DB will be published with the v1.0 release; for now, work against your own).

### Clone and install

```bash
git clone https://github.com/<your-org>/seedsoftruth.git
cd seedsoftruth
git checkout -b feat/your-change

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

Copy the env-var template (once it lands as part of the v1.0 release; for now, see [CONFIG.md](CONFIG.md) for the full reference):

```bash
cp .env.example .env   # planned
# edit .env with FLASK_SECRET_KEY, SOT_PASSWORDS, MODEL_ADAPTER, and at least one backend credential
```

For retrieval-only work you can use the `sim` adapter, which returns deterministic stub responses without making any network calls:

```bash
export MODEL_ADAPTER=sim
```

### Run locally

```bash
flask --app app run
```

Open http://localhost:5000.

## Running tests

> **Honest status:** the test suite is currently a set of shell scripts in `scripts/test/` that hit a running server, plus two Python smoke tests. A pytest suite is one of the highest-priority items on the v1.0 roadmap. Contributing a pytest harness or porting one of the existing shell tests is welcome and will be merged quickly.

For now, end-to-end smoke tests against a running local server:

```bash
# Start the server in one terminal
flask --app app run

# In another terminal
./scripts/test/test_health_check.sh
./scripts/test/test_unlock.sh
./scripts/test/test_search_req.sh
```

Retrieval evaluation (compares ranking-algorithm variants on a labeled query set):

```bash
./scripts/run_eval.sh
```

When the pytest suite lands, the canonical command will be `pytest`. Until then, please describe in your PR what manual tests you ran.

## Code style

> **Status:** lint and type-check tooling is being introduced as part of the v1.0 cleanup. Until they are wired into CI, please follow the existing style in the file you are editing.

Conventions in use today:

- **Python**: 4-space indent, type hints on public functions, double-quoted strings, f-strings for user-facing messages and `%s`-style for logger calls. Module-level docstrings explain purpose; function docstrings explain non-obvious behavior.
- **JavaScript** (`static/app.js`): vanilla JS, ES2017+, 2-space indent. No build step.
- **SQL**: lowercase keywords, snake_case identifiers, multi-line for any query with more than two clauses.

Once `ruff` + `black` + `mypy` land in CI, this section will be replaced with `make lint` / `make typecheck`.

## Commit messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short summary>

<optional body explaining the why, not the what>

<optional footer with issue refs, BREAKING CHANGE notes, etc.>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`.

Examples:

```
feat(rag): add embedding layer alongside BM25 hybrid retrieval

fix(auth): stop logging plaintext password in /api/unlock

docs(readme): clarify corpus schema requirements

refactor(rag_controller): split retrieval pipeline into a package
```

Keep commits focused; one logical change per commit. If a PR contains multiple commits, they should each compile and pass tests on their own.

## Pull request process

1. **Fork and branch.** Branch off `main`. Use a descriptive branch name like `feat/embeddings-layer` or `fix/csrf-on-chat`.
2. **Open the PR early.** Mark it as draft if it isn't ready for review. Early PRs let us flag direction problems before you invest more time.
3. **Reference issues.** If your PR closes an issue, include `Closes #<n>` in the description.
4. **Describe the change.** Three paragraphs in the PR body: *what* the change does, *why* it is needed, and *how* you tested it.
5. **Keep PRs reviewable.** Smaller is better. ~300 lines of diff is the ceiling we can review quickly; bigger changes need a heads-up.
6. **Respond to review.** A maintainer will respond within ~3 working days. If review comments require substantive rework, we will re-review the rework — no need to apologize.
7. **Squash on merge.** PRs are squashed into a single commit on `main`. Your commit messages on the branch don't need to be perfect; the squashed message will follow the conventional-commits format above.

## Reporting bugs

Open a GitHub issue with the **bug report** template. At minimum include:

- What you expected to happen
- What actually happened
- A minimal reproduction (URL, payload, or commands)
- Relevant log output (with secrets redacted)
- Your Python version, OS, and how you are running the app (gunicorn / flask dev server)

If the bug is a security vulnerability, **do not open a public issue**. See [SECURITY.md](SECURITY.md) for the disclosure channel.

## Suggesting features

Open a GitHub issue with the **feature request** template. Tell us:

- What problem you are trying to solve (the underlying need, not a proposed solution)
- Who else has this problem (it helps us prioritize)
- Any prior art or alternatives you have considered

We are especially interested in features that come with a willingness to implement them. "I'd like X and I can put together a PR if you think it fits" is the most actionable kind of feature request.

## Questions

If you are unsure whether something belongs in the project, open a GitHub Discussion or a draft issue. We would rather have a five-minute conversation up front than spend a week building the wrong thing.
