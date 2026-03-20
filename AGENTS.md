# AGENTS.md

## Implementation
- Avoid silent failures. Unexpected states, violated assumptions, and invalid inputs should raise clear errors rather than quietly continuing with incorrect behavior.
- Prefer minimal, localized diffs unless a broader refactor is clearly necessary for correctness or maintainability.
- Do not rename public APIs, change file formats, or alter external interfaces unless necessary.
- Run tests for touched modules when relevant tests exist. If tests cannot be run, state that explicitly.
- Keep tensor operations vectorized and avoid Python loops in performance-sensitive paths unless there is a clear reason not to.
- Prefer complete fixes over partial patches. When appropriate, update the implementation, tests, and closely related call sites together.
- Do not introduce `import ast` or use `ast.literal_eval` unless explicitly justified. When the input format is known, write a simple explicit parser instead.
- Keep data transformations explicit, readable, and easy to verify.
- Parallelize independent expensive work when it is safe, deterministic, and worth the added complexity.
- Do not sacrifice determinism, reproducibility, or result ordering for speed unless explicitly requested.
- Do not remove existing `pdb.set_trace()` calls added by the user unless explicitly asked.
- Prefer straightforward implementations over clever or heavily abstracted ones unless the abstraction materially improves correctness or maintainability.

## Communication
- Be concise and direct.
- Make the requested change when possible, rather than only describing what could be done.
- When behavior changes, state the practical downstream effect.
- Call out anything you could not verify, run, or complete.
