"""Interactive prompts. Re-prompt on invalid input rather than erroring out."""
from typing import List, Optional


def ask_choice(prompt: str, choices: List[str], allow_short_index: bool = True) -> str:
    """Print numbered choices, accept either the number or the name."""
    print(prompt)
    for i, c in enumerate(choices, 1):
        print(f"  [{i}] {c}")
    while True:
        raw = input("> ").strip().lower()
        if not raw:
            print("Please make a choice.")
            continue
        if allow_short_index and raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"Out of range. Pick 1-{len(choices)}.")
            continue
        for c in choices:
            if raw == c.lower():
                return c
        print(f"Unknown choice {raw!r}. Pick a number or one of: {choices}")


def ask_text(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw: return raw
        if default is not None: return default
        print("Please enter a value.")


def ask_severity(grid: List[int]) -> int:
    print(f"Choose severity ({min(grid)}-{max(grid)})")
    while True:
        raw = input("> ").strip()
        if raw.isdigit() and int(raw) in grid:
            return int(raw)
        print(f"Severity must be one of {grid}.")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    s = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} ({s}): ").strip().lower()
        if not raw: return default
        if raw in ("y", "yes"): return True
        if raw in ("n", "no"):  return False


def ask_columns(prompt: str, available: List[str], allow_all: bool = True,
                multi: bool = True) -> List[str]:
    """Pick one or more columns from the available set."""
    print(prompt)
    print("Available columns: " + ", ".join(available))
    if allow_all:
        print("(Enter 'all' to select every column, or comma-separated names)")
    while True:
        raw = input("> ").strip()
        if not raw:
            print("Please enter at least one column.")
            continue
        if allow_all and raw.lower() == "all":
            return list(available)
        picks = [p.strip() for p in raw.split(",")]
        bad = [p for p in picks if p not in available]
        if bad:
            print(f"Unknown columns: {bad}. Pick from: {available}")
            continue
        if not multi and len(picks) > 1:
            print("Please pick exactly one column.")
            continue
        return picks
