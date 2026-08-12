# Packaging and cleanup plan

## Current status

- Core RAG runtime is working with a reusable `RAGEngine`.
- REPL chat flow is available via `scripts/rag_chat.py`.
- Tests are passing and the repo is close to packaging readiness.

## Next steps

1. Finalize packaging metadata (`setup.py`, runtime dependencies, console entry points).
2. Add package-level script support and verify `pip install -e .` works.
3. Confirm import paths for `scripts.*` and CLI entry points.
4. Re-run focused validation after the packaging changes.
5. Consider splitting the current monolithic script layer into clearer library entry points later.
