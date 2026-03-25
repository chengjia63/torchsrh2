# AGENTS.md

## Implementation
- Avoid silent failures. Unexpected states, violated assumptions, and invalid inputs should raise clear errors rather than quietly continuing with incorrect behavior.
- Prefer minimal, localized diffs unless a broader refactor is clearly necessary for correctness or maintainability.
- Do not rename public APIs, change file formats, or alter external interfaces unless necessary.
- Preserve existing interfaces exactly unless the user explicitly asks to change them. This includes dataset return keys, config structure, file layout, and logging assumptions relied on elsewhere.
- Do not modify files the user has said are out of scope or not authorized. If a fix appears to require such a file, stop and ask instead of editing around it silently.
- Run tests for touched modules when relevant tests exist. If tests cannot be run, state that explicitly.
- Keep tensor operations vectorized and avoid Python loops in performance-sensitive paths unless there is a clear reason not to.
- Prefer complete fixes over partial patches. When appropriate, update the implementation, tests, and closely related call sites together.
- Do not introduce `import ast` or use `ast.literal_eval` unless explicitly justified. When the input format is known, write a simple explicit parser instead.
- Keep data transformations explicit, readable, and easy to verify.
- Before patching, verify the concrete local call path and data/schema assumptions in the current files instead of inferring architecture from memory.
- Parallelize independent expensive work when it is safe, deterministic, and worth the added complexity.
- Do not sacrifice determinism, reproducibility, or result ordering for speed unless explicitly requested.
- Do not remove existing `pdb.set_trace()` calls added by the user unless explicitly asked.
- Do not report user-added `pdb.set_trace()` calls as bugs or cleanup items unless the user explicitly asks for review of them.
- Prefer straightforward implementations over clever or heavily abstracted ones unless the abstraction materially improves correctness or maintainability.
- Avoid adding speculative abstractions, feature-detection branches, marker attributes, or defensive helper layers unless they are required for correctness.

## Communication
- Be concise and direct.
- Make the requested change when possible, rather than only describing what could be done.
- When the user asks for a direct fix, prefer editing the code immediately over presenting options or long explanations.
- When behavior changes, state the practical downstream effect.
- Call out anything you could not verify, run, or complete.
