---
applyTo: "**"
---

# Project Context

- This is a Python RAG application using SentenceTransformers, FAISS, FastAPI, Streamlit, and pytest.
- Favor simple, maintainable solutions consistent with established Python practices.
- Follow the existing project structure unless there is a clear reason to change it.

# Coding

- Prefer straightforward implementations over unnecessary abstractions.
- Make the smallest coherent change needed for the task.
- Do not modify unrelated code or add dependencies without a clear benefit.
- During refactoring, preserve existing behavior and public interfaces unless a change is explicitly requested.
- Make sure to run tests after changes to verify that behavior is preserved.

# Testing

- Use pytest.
- Add or update tests when relevant.
- Prefer test names in the form:
  `test_<function>_WHEN_<condition>_THEN_<expected_behavior>`.
- Test observable behavior and important edge cases without unnecessary duplication.

# Documentation

- Follow PEP 257 and use NumPy-style docstrings.
- Keep docstrings and comments concise.
- Document non-obvious behavior; do not restate the code.
- Do not expose personal or sensitive information in documentation or comments.

# Working Style

- Inspect the relevant implementation and tests before making changes.
- After changes, identify relevant tests to run.
- Keep explanations concise and focused on:
  1. What changed.
  2. Why.
  3. What to do next, if anything.

# Recommendations

When comparing improvement options:

- Clearly distinguish the meaningful alternatives.
- Give up to three relevant pros and cons for each.
- Compare them directly using relevant criteria such as simplicity, maintainability, behavior, performance, or testability.
- Recommend one option, preferring the simplest solution that adequately solves the current problem.
