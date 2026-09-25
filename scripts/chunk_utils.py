"""
chunk_utils.py

Shared chunking helpers for artifacts that exceed GitHub's 100MB file limit.
Files are committed as `{stem}_chunk_NNN{ext}` byte parts and reassembled on
demand, so a fresh clone is runnable without Git LFS.

This is the same module used by the sibling project e85, kept in sync so both
repos behave identically. The chunk format is plain byte concatenation and is
unchanged from the original version of this file -- verified bidirectionally
against the 344 committed vjepa2 checkpoint chunks, so upgrading here does not
invalidate anything already in the repo.

Use these instead of bare np.load / torch.load / Path.exists on any artifact
that may be chunked: the fused file does NOT exist in a fresh clone, so a raw
loader silently finds nothing.

  checkpoints (.pt) : save_chunked / load_chunked
  arrays (.npz)     : save_npz / load_npz / npz_exists / discover_npz
  image zips        : ensure_fused_zip
  any file          : chunk_file_if_needed / fuse_chunks_to_file
"""

import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

# GitHub hard-blocks files > 100MB. Stay a bit under for safety.
GITHUB_MAX_BYTES = 100 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 95 * 1024 * 1024  # npz / large binary payloads
DEFAULT_PT_CHUNK_BYTES = 23 * 1024 * 1024  # checkpoints (legacy <25MB chunks)

_CHUNK_STEM_RE = re.compile(r"^(.*)_chunk_(\d{3})$")


def get_chunk_paths(base_path):
    """Get sorted list of chunk files for a given base path."""
    base_path = Path(base_path)
    parent = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix

    # Locate all files matching base_name_chunk_*.suffix
    pattern = f"{stem}_chunk_*{suffix}"
    chunks = sorted(parent.glob(pattern))
    return chunks


def check_chunked_exists(base_path):
    """Check if chunked files exist for the given base path."""
    base_path = Path(base_path)
    first_chunk = base_path.parent / f"{base_path.stem}_chunk_000{base_path.suffix}"
    return first_chunk.exists()


def npz_exists(base_path):
    """True if a single .npz or its chunk set is present."""
    base_path = Path(base_path)
    return base_path.exists() or check_chunked_exists(base_path)


def is_chunk_file(path):
    """True if path looks like name_chunk_000.ext."""
    return bool(_CHUNK_STEM_RE.match(Path(path).stem))


def logical_npz_path(path):
    """
    Map a chunk path back to its logical base (.npz), or return path unchanged.
    e.g. foo_chunk_003.npz -> foo.npz
    """
    path = Path(path)
    m = _CHUNK_STEM_RE.match(path.stem)
    if m:
        return path.parent / f"{m.group(1)}{path.suffix}"
    return path


def clean_existing_chunks(base_path):
    """Delete any existing chunk files matching the pattern."""
    for chunk in get_chunk_paths(base_path):
        try:
            if chunk.exists():
                chunk.unlink()
        except Exception as e:
            print(f"Warning: failed to delete old chunk {chunk}: {e}")


def discover_npz(directory, recursive=False):
    """
    Discover logical .npz bases in a directory, treating chunk sets as one entry.
    Returns sorted Paths pointing at the logical base name (may not exist on disk
    if only chunks are present).
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    iterator = directory.rglob("*.npz") if recursive else directory.glob("*.npz")
    bases = set()
    for p in iterator:
        bases.add(logical_npz_path(p))
    return sorted(bases)


def _split_file_into_chunks(source_file, base_path, chunk_size):
    """Byte-split source_file into base_stem_chunk_NNN.suffix next to base_path."""
    base_path = Path(base_path)
    source_file = Path(source_file)
    file_size = source_file.stat().st_size
    print(
        f"Splitting {source_file.name} ({file_size / 1e6:.2f} MB) "
        f"into chunks of {chunk_size / 1e6:.2f} MB..."
    )

    chunk_idx = 0
    with open(source_file, "rb") as f_in:
        while True:
            data = f_in.read(chunk_size)
            if not data:
                break
            chunk_name = f"{base_path.stem}_chunk_{chunk_idx:03d}{base_path.suffix}"
            chunk_path = base_path.parent / chunk_name
            with open(chunk_path, "wb") as f_out:
                f_out.write(data)
            chunk_idx += 1

    print(f"Successfully wrote {chunk_idx} chunk files for {base_path.name}")
    return chunk_idx


def fuse_chunks_to_file(base_path, dest_path):
    """Concatenate chunk files for base_path into dest_path."""
    base_path = Path(base_path)
    chunks = get_chunk_paths(base_path)
    if not chunks:
        raise FileNotFoundError(f"No chunked files found for {base_path}")
    with open(dest_path, "wb") as f_out:
        for chunk_path in chunks:
            with open(chunk_path, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
    return len(chunks)


def chunk_file_if_needed(path, chunk_size=DEFAULT_CHUNK_BYTES):
    """
    If path exceeds chunk_size, split into byte chunks and delete the original.
    Returns list of paths that represent the payload (single file or chunks).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= chunk_size:
        return [path]

    clean_existing_chunks(path)
    _split_file_into_chunks(path, path, chunk_size=chunk_size)
    path.unlink()
    return get_chunk_paths(path)


def save_chunked(state_dict_or_path, base_path, chunk_size=DEFAULT_PT_CHUNK_BYTES):
    """
    Saves a state dict (or slices an existing file) into chunks < 25MB.

    Args:
        state_dict_or_path: OrderedDict/dict of state dict, or str/Path to an existing .pt file.
        base_path: The target base checkpoint path (e.g. /path/to/model.pt).
        chunk_size: Size in bytes of each chunk. Default is 23MB (strictly < 25MB).
    """
    import torch

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # First, clean existing chunks to avoid leftover files from previous runs
    clean_existing_chunks(base_path)

    # If state_dict_or_path is already a file, we can read from it directly.
    # Otherwise, we serialize the state_dict to a temp file first.
    temp_filepath = None
    if isinstance(state_dict_or_path, (str, Path)):
        source_file = Path(state_dict_or_path)
    else:
        # Create a temp file to serialize the state dict
        fd, temp_filepath = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        source_file = Path(temp_filepath)
        print(f"Serializing state_dict to temporary file: {source_file}")
        torch.save(state_dict_or_path, source_file)

    try:
        _split_file_into_chunks(source_file, base_path, chunk_size=chunk_size)
    finally:
        # Clean up temp file if we created one
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass


def load_chunked(base_path, map_location="cpu"):
    """
    Finds chunks, fuses them into a temporary file on disk, loads the state dict, and cleans up.

    Args:
        base_path: The target base checkpoint path (e.g. /path/to/model.pt).
        map_location: Where to load tensors (default 'cpu').
    """
    import torch

    base_path = Path(base_path)
    chunks = get_chunk_paths(base_path)
    if not chunks:
        raise FileNotFoundError(f"No chunked files found for {base_path}")

    print(f"Fusing {len(chunks)} chunks for {base_path.name}...")

    # Create temp file
    fd, temp_filepath = tempfile.mkstemp(suffix=".pt")
    os.close(fd)

    try:
        fuse_chunks_to_file(base_path, temp_filepath)
        print("Loading state dict from fused temp file...")
        state_dict = torch.load(temp_filepath, map_location=map_location)
        return state_dict
    finally:
        # Clean up temp file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass


def save_npz(path, max_bytes=DEFAULT_CHUNK_BYTES, **arrays):
    """
    Write a compressed .npz. If the result exceeds max_bytes (GitHub 100MB limit),
    split into byte chunks named {stem}_chunk_NNN.npz and remove the oversized file.

    Returns the list of paths that remain on disk (single file or chunks).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_existing_chunks(path)
    if path.exists():
        path.unlink()

    np.savez_compressed(path, **arrays)
    size = path.stat().st_size
    if size <= max_bytes:
        print(f"Saved {path.name} ({size / 1e6:.1f} MB)")
        return [path]

    print(
        f"{path.name} is {size / 1e6:.1f} MB "
        f"(>{max_bytes / 1e6:.0f} MB GitHub limit); chunking..."
    )
    return chunk_file_if_needed(path, chunk_size=max_bytes)


def load_npz(path):
    """
    Load a .npz by logical path. Accepts either a single file or a chunk set
    ({stem}_chunk_000.npz, ...). Returns a plain dict of arrays (materialized so
    temp fuse files can be deleted immediately).
    """
    path = Path(path)
    path = logical_npz_path(path)

    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    chunks = get_chunk_paths(path)
    if not chunks:
        raise FileNotFoundError(f"No .npz or chunks found for {path}")

    print(f"Fusing {len(chunks)} chunks for {path.name}...")
    fd, temp_filepath = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        fuse_chunks_to_file(path, temp_filepath)
        with np.load(temp_filepath, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    finally:
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass


# ── Shared preds / zip schema helpers ─────────────────────────────────────────
# Canonical general-population format (merged FairFace+FFHQ, FFHQ bulk, etc.):
#   one .npz per image with preds shape (20484,) and key 'filename' (str)
# Target / legacy multi-image format still accepted:
#   preds (N, 20484) + 'filenames' (N,)


def preds_as_image_vectors(preds):
    """Normalize preds to (N_images, 20484). Never mean a 1D vector across vertices."""
    preds = np.asarray(preds)
    if preds.ndim == 1:
        if preds.shape[0] != 20484:
            raise ValueError(f"unexpected 1D preds shape {preds.shape}")
        return preds.reshape(1, -1)
    if preds.ndim == 2 and preds.shape[1] == 20484:
        return preds
    raise ValueError(f"unexpected preds shape {preds.shape}")


def npz_image_names(data):
    """
    Return list[str] of zip-member / image names from an npz dict.
    Accepts singular 'filename' (per-image) or plural 'filenames' (multi-image).
    """
    if "filenames" in data:
        arr = np.asarray(data["filenames"])
        return [str(x) for x in arr.tolist()]
    if "filename" in data:
        v = data["filename"]
        if isinstance(v, np.ndarray):
            v = v.item() if v.ndim == 0 else str(v.tolist())
        return [str(v)]
    raise KeyError("npz missing 'filename' or 'filenames'")


def ensure_fused_zip(base_path, dest_path=None):
    """
    Return a path to a usable .zip.

    If base_path already exists, return it. Otherwise look for
    {stem}_chunk_NNN{suffix} next to it, fuse them into dest_path
    (default: base_path), and return that.
    """
    base_path = Path(base_path)
    dest_path = Path(dest_path) if dest_path is not None else base_path
    if base_path.exists():
        return base_path
    if dest_path.exists() and dest_path != base_path:
        return dest_path
    chunks = get_chunk_paths(base_path)
    if not chunks:
        raise FileNotFoundError(
            f"No zip at {base_path} and no chunks "
            f"({base_path.stem}_chunk_NNN{base_path.suffix})"
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fusing {len(chunks)} zip chunks -> {dest_path} ...")
    fuse_chunks_to_file(base_path, dest_path)
    return dest_path


def resolve_zip_member(zf, name):
    """
    Resolve an image name against a zip's namelist.
    Tries exact match, then basename-only match if unique.
    """
    names = zf.namelist()
    if name in names:
        return name
    base = Path(name).name
    hits = [n for n in names if Path(n).name == base]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise KeyError(f"ambiguous zip member basename {base!r}: {hits[:5]}...")
    raise KeyError(f"zip member not found: {name!r}")
