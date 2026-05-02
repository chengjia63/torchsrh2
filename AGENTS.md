# AGENTS.md



## Implementation
- Do not run Python, execute scripts, or start servers. Make the code changes and stop; leave running and verification to the user.
- Avoid silent failures: do not catch, mask, log-and-continue, or substitute defaults for errors without telling the user.
- Do not add validation for conditions that would already cause an immediate, clear crash downstream — a missing key, a None where an array is required, a missing file. The natural error is sufficient and an added assert or raise only clutters the diff.
- Do not allow a wrong or missing input to pass through silently. If the code would otherwise continue past a real problem (wrong shape, wrong dtype, silently-skipped branch, swallowed exception, default-substituted value), that is not acceptable — surface it.
- Prefer minimal, localized diffs unless a broader refactor is clearly necessary for correctness or maintainability.
- Do not rename public APIs, change file formats, or alter external interfaces unless necessary.
- Preserve existing interfaces exactly unless the user explicitly asks to change them. This includes dataset return keys, config structure, file layout, and logging assumptions relied on elsewhere.
- When the user asks to rename a variable, config key, or public argument, make it a clean rename without backward-compatible aliases unless they explicitly ask for compatibility.
- Do not modify files the user has said are out of scope or not authorized. If a fix appears to require such a file, stop and ask instead of editing around it silently.
- Keep tensor operations vectorized and avoid Python loops in performance-sensitive paths unless there is a clear reason not to.
- Fix the root cause, not the symptom. Do not patch around the issue, suppress the error, or add a workaround that makes the failure go away without addressing what caused it. If the correct fix is larger than expected, do that fix rather than a smaller one that only technically satisfies the request.
- Do not introduce `import ast` or use `ast.literal_eval` unless explicitly justified. When the input format is known, write a simple explicit parser instead.
- Do not use `pathlib`; use `os` and `os.path` for filesystem paths.
- Keep data transformations explicit, readable, and easy to verify.
- Before patching, verify the concrete local call path and data/schema assumptions in the current files instead of inferring architecture from memory.
- Parallelize independent expensive work when it is safe, deterministic, and worth the added complexity.
- Do not sacrifice determinism, reproducibility, or result ordering for speed unless explicitly requested.
- Do not remove existing `pdb.set_trace()` calls added by the user unless explicitly asked.
- Do not report user-added `pdb.set_trace()` calls as bugs or cleanup items unless the user explicitly asks for review of them.
- Prefer straightforward implementations over clever or heavily abstracted ones unless the abstraction materially improves correctness or maintainability.
- Avoid adding speculative abstractions, feature-detection branches, marker attributes, validation layers, or defensive helper layers unless they are required for correctness.
- Include type hints on all new and modified code. Preserve existing type hints; do not remove or weaken them (e.g. replacing a concrete type with Any).
- Respect the existing dtype, axis order, and device of input arrays and tensors. Do not silently upcast (e.g. float32 → float64), reorder axes (ZYX vs XYZ, channels-first vs channels-last), or move data between CPU and GPU unless the requested behavior requires it.
- Do not introduce new RNG calls without seeding them through the existing pattern used in the surrounding code.
- Do not add new dependencies unless necessary and explicitly approved.
- Do not create, move, or delete files unless required by the requested change. Do not rename files unless explicitly requested.
- Do not add explanatory comments unless they clarify non-obvious behavior that cannot be made clear in code.
- Never print, modify, or commit secrets, tokens, credentials, `.env` files, or private keys.
- Do not add telemetry, network calls, or external service dependencies unless explicitly requested.
- If the user asks for review, critique, or diagnosis, do not edit unless they explicitly ask for a fix. If the user asks for a direct fix, edit immediately.
- The user can be wrong. If the requested change conflicts with the concrete code path, data/schema assumptions, public interfaces, reproducibility, or the likely root cause, push back directly with evidence instead of implementing an incorrect change.
- The user can be wrong, even when they state something with confidence. If the requested change or stated assumption conflicts with the concrete code path, data/schema assumptions, public interfaces, reproducibility, or the likely root cause, push back directly with evidence instead of implementing an incorrect change.

## Communication
- Be concise and direct.
- Make the requested change when possible, rather than only describing what could be done.
- When the user asks for a direct fix, prefer editing the code immediately over presenting options or long explanations.
- When behavior changes, state the practical downstream effect.
- Call out anything you could not verify, run, or complete.
- Do not restate the request back before answering. Skip preambles like "Great question" or "You're right that…" and get to the response.
- When blocked by a missing file, ambiguous spec, or unclear intent, stop and ask. Do not guess and proceed.
- Distinguish what you did from what you would do. Do not describe planned or hypothetical changes in the past tense, and do not claim a change was made unless the edit was actually applied.
