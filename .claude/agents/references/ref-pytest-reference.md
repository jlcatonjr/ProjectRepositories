<!-- AGENTTEAMS:BEGIN content v=1 -->
# pytest Reference — ProjectRepositories

> Quick-reference for **pytest 7.1.2** (library) in ProjectRepositories.
> This is a lightweight reference file, not an agent. For operational procedures, consult the tool's reference/skill document, or escalate to `@orchestrator`.

---

## Version

`pytest` `7.1.2`

## Configuration

**Config files:** `N/A`

## Official Documentation

https://docs.pytest.org/en/stable/

## Key API Surface

- **Test discovery** — files matching `test_*.py` or `*_test.py`; functions prefixed `test_`
- **`pytest.fixture(scope=...)`** — `"function"` (default), `"class"`, `"module"`, `"session"`
- **`pytest.mark`** — `@pytest.mark.parametrize(argnames, argvalues)`, `@pytest.mark.skip`, `@pytest.mark.skipif`, `@pytest.mark.xfail`
- **`monkeypatch`** — `.setattr()`, `.setenv()`, `.delenv()`, `.setitem()`, `.chdir()`
- **`tmp_path`** — `pathlib.Path` fixture scoped to the test function
- **`capsys` / `capfd`** — capture stdout/stderr; `.readouterr()` returns `(out, err)`
- **`conftest.py`** — project-wide fixtures; auto-used by all tests in the directory
- **Exit codes** — 0: all passed, 1: some failed, 2: interrupted, 3: internal error, 4: bad usage, 5: no tests

<!-- Document the primary classes, functions, or APIs that project code depends on from pytest. -->

## Common Patterns & Pitfalls

- **Fixtures over setUp/tearDown** — pytest fixtures are composable and scopeable; prefer them over `unittest.TestCase` methods
- **`parametrize` flattens test matrices** — `@pytest.mark.parametrize("a,b,expected", [...])` generates one test per row; ID is auto-derived
- **`conftest.py` for shared fixtures** — never duplicate fixtures across test files; put project-wide fixtures in `tests/conftest.py`
- **`tmp_path` not `tempfile`** — `tmp_path` is auto-cleaned and returns a `pathlib.Path`; no teardown needed
- **`monkeypatch` scope** — only valid within the test function; never use it in session-scoped fixtures unless you understand the implications
- **`-x` stops on first failure** — use during development; CI should run without `-x` to see all failures
- **`-q` for clean output** — `python -m pytest tests/ -q` suppresses verbose per-test output in CI logs

<!-- Document common usage patterns, best practices, and known issues for pytest 7.1.2. -->

## Key Conventions

- Follow project style rules when using pytest
- Refer to authority sources for API contract accuracy
- Validate changes against existing tests before committing

## Related Agents

- `@technical-validator` — verify technical accuracy of pytest usage
- `@primary-producer` — implements code that depends on pytest
<!-- AGENTTEAMS:END content -->
