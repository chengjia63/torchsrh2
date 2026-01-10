#!/usr/bin/env python3
import ast
import json
import sys
from pathlib import Path

def _find_matching_bracket(s: str, start_idx: int) -> int:
    """
    Given s[start_idx] == '[', return index of the matching ']'.
    Handles strings and escapes.
    Returns -1 if not found (e.g., truncated).
    """
    assert s[start_idx] == "["
    depth = 0
    i = start_idx
    in_str = False
    esc = False
    quote = ""
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1

def recover_code_sources(text: str):
    sources = []
    # Find each code cell marker, then find the next "source": [ ... ] after it.
    key = '"cell_type"'
    i = 0
    while True:
        i = text.find(key, i)
        if i == -1:
            break
        # cheap check for code cell
        j = text.find('"code"', i, i + 200)  # usually near cell_type
        if j == -1:
            i += len(key)
            continue

        # find "source" after this cell_type
        sidx = text.find('"source"', j)
        if sidx == -1:
            i = j + 6
            continue

        # find the '[' starting the list
        lb = text.find("[", sidx)
        if lb == -1:
            i = sidx + 8
            continue

        rb = _find_matching_bracket(text, lb)
        if rb == -1:
            # truncated list; stop scanning further (file likely ends mid-cell)
            break

        src_list_txt = text[lb : rb + 1]
        # parse list of strings
        try:
            src_list = json.loads(src_list_txt)
        except Exception:
            # tolerate some JSON quirks by falling back to python literal parsing
            src_list = ast.literal_eval(src_list_txt)

        if isinstance(src_list, list) and all(isinstance(x, str) for x in src_list):
            sources.append("".join(src_list))

        i = rb + 1
    return sources

def main(ipynb_path: str):
    text = Path(ipynb_path).read_text(encoding="utf-8", errors="replace")

    # quick sanity/debug counts
    n_code_markers = text.count('"cell_type"')  # rough
    n_code = text.count('"cell_type": "code"')
    print(f'Found occurrences: cell_type={n_code_markers}, code_cells_exact={n_code}')

    sources = recover_code_sources(text)
    if not sources:
        raise SystemExit("No code sources extracted. If you paste the first ~300 lines of the file, I can adapt further.")

    # write .py
    py_blocks = []
    for k, code in enumerate(sources, 1):
        py_blocks.append(f"# %% [cell {k}]\n{code.rstrip()}\n")
    Path("recovered_cells.py").write_text("\n".join(py_blocks), encoding="utf-8")

    # write minimal ipynb
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
            for code in sources
        ],
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path("recovered.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    print(f"Recovered {len(sources)} code cells")
    print("Wrote: recovered_cells.py")
    print("Wrote: recovered.ipynb")

if __name__ == "__main__":
    main(sys.argv[1])
