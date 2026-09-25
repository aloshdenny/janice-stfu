"""
verify_reproducible.py

Proves a fresh clone of this repo is actually runnable: every oversized artifact
ships as chunks, every chunk set present really re-fuses, and no script reaches
for a chunked checkpoint with a raw torch.load that would not find it.

WHY THIS EXISTS. The abliterated vjepa2 checkpoints exceed GitHub's 100MB limit,
so they are committed as `{stem}_chunk_NNN.pt` parts via chunk_utils.save_chunked
and must be read back with chunk_utils.load_chunked. The fused .pt does NOT exist
in a fresh clone, so a bare `torch.load(ckpt)` or `ckpt.exists()` silently fails
or skips the data rather than erroring usefully.

Run from the repo root:
  python scripts/verify_reproducible.py
Exit code 0 = reproducible, 1 = something would break for a fresh cloner.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import chunk_utils

REPO = Path(__file__).resolve().parent.parent
GITHUB_HARD_LIMIT = 100 * 1024 * 1024
CHUNK_RE = re.compile(r"^(.*)_chunk_(\d{3})(\..+)$")

failures, skipped = [], []


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    return [REPO / f for f in out.stdout.splitlines() if f]


def check_no_oversized(files):
    print("\n[1] no committed file exceeds GitHub's 100MB hard limit")
    present = [f for f in files if f.is_file()]
    bad = [f for f in present if f.stat().st_size > GITHUB_HARD_LIMIT]
    for f in bad:
        failures.append(f"{f.relative_to(REPO)} is {f.stat().st_size/1e6:.0f}MB (>100MB)")
    print(f"    {len(present)} files present locally, {len(bad)} oversized")


def check_chunk_sets(files):
    print("\n[2] every chunk set present re-fuses")
    bases = set()
    for f in files:
        m = CHUNK_RE.match(f.name)
        if m:
            bases.add(f.parent / (m.group(1) + m.group(3)))
    if not bases:
        print("    no chunked artifacts found")
    for base in sorted(bases):
        rel = base.relative_to(REPO)
        parts = chunk_utils.get_chunk_paths(base)
        if not parts:
            # Tracked but not checked out: normal in the partial/sparse clone the
            # README recommends. Not a defect -- only a present artifact that fails
            # to reconstruct is a real failure.
            skipped.append(str(rel))
            print(f"    SKIP {rel} (not checked out in this clone)")
            continue
        try:
            sd = chunk_utils.load_chunked(base, map_location="cpu")
            n = len(sd) if hasattr(sd, "__len__") else "?"
            print(f"    OK   {rel}  ({len(parts)} parts -> {n} tensors)")
        except Exception as e:
            failures.append(f"{rel}: chunk set does not reconstruct ({e})")


def check_loaders(files):
    """Flag a raw loader only when applied to a COMMITTED chunked artifact.

    Precision over recall: a check that cries wolf gets ignored, so this
    resolves which variables actually hold a chunked path before flagging.
    """
    print("\n[3] no script uses a raw loader on a committed chunked artifact")
    stems = set()
    for f in files:
        m = CHUNK_RE.match(f.name)
        if m:
            stems.add(m.group(1) + m.group(3))

    raw_call = re.compile(r"\b(torch\.load|torch\.save|np\.load|np\.savez)\s*\(\s*([A-Za-z_][\w\.\[\]\"']*)")
    raw_exists = re.compile(r"([A-Za-z_][\w\.]*)\.exists\s*\(\s*\)")
    flagged = 0

    for py in sorted((REPO / "scripts").glob("*.py")):
        if py.name in ("chunk_utils.py", "verify_reproducible.py"):
            continue
        lines = py.read_text(errors="ignore").splitlines()
        bound = set()
        for line in lines:
            if any(s in line for s in stems) and "=" in line:
                lhs = line.split("=", 1)[0].strip()
                if re.fullmatch(r"[A-Za-z_]\w*", lhs):
                    bound.add(lhs)
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                continue
            targets = []
            m = raw_call.search(line)
            if m:
                targets.append((m.group(1), m.group(2)))
            m2 = raw_exists.search(line)
            if m2:
                targets.append((".exists()", m2.group(1)))
            for call, arg in targets:
                base = arg.split(".")[0].split("[")[0].strip("\"'")
                if base in bound or any(s in line for s in stems):
                    flagged += 1
                    failures.append(
                        f"{py.name}:{i} `{call}` on chunked artifact `{base}` "
                        f"-> use chunk_utils.load_chunked / save_chunked")
    print(f"    {len(stems)} committed chunked artifact(s); {flagged} raw-loader use(s)")


def main():
    files = tracked_files()
    check_no_oversized(files)
    check_chunk_sets(files)
    check_loaders(files)
    print("\n" + "=" * 62)
    for f in failures:
        print(f"  FAIL  {f}")
    if failures:
        print(f"\nNOT REPRODUCIBLE: {len(failures)} blocking issue(s)")
        return 1
    tail = ""
    if skipped:
        tail = (f"; {len(skipped)} artifact(s) not checked out in this clone, "
                f"run `git sparse-checkout disable` to fetch and verify them")
    print(f"\nREPRODUCIBLE: chunk sets reconstruct, nothing exceeds the size limit{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
