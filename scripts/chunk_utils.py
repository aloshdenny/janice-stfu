import os
import shutil
import tempfile
import torch
from pathlib import Path

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

def clean_existing_chunks(base_path):
    """Delete any existing chunk files matching the pattern."""
    for chunk in get_chunk_paths(base_path):
        try:
            if chunk.exists():
                chunk.unlink()
        except Exception as e:
            print(f"Warning: failed to delete old chunk {chunk}: {e}")

def save_chunked(state_dict_or_path, base_path, chunk_size=23 * 1024 * 1024):
    """
    Saves a state dict (or slices an existing file) into chunks < 25MB.
    
    Args:
        state_dict_or_path: OrderedDict/dict of state dict, or str/Path to an existing .pt file.
        base_path: The target base checkpoint path (e.g. /path/to/model.pt).
        chunk_size: Size in bytes of each chunk. Default is 23MB (strictly < 25MB).
    """
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
        # Split source_file into chunks
        file_size = source_file.stat().st_size
        print(f"Splitting {source_file.name} ({file_size / 1e6:.2f} MB) into chunks of {chunk_size / 1e6:.2f} MB...")
        
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
    base_path = Path(base_path)
    chunks = get_chunk_paths(base_path)
    if not chunks:
        raise FileNotFoundError(f"No chunked files found for {base_path}")
        
    print(f"Fusing {len(chunks)} chunks for {base_path.name}...")
    
    # Create temp file
    fd, temp_filepath = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    
    try:
        # Concatenate chunks into temp file
        with open(temp_filepath, "wb") as f_out:
            for chunk_path in chunks:
                with open(chunk_path, "rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
                    
        # Load state dict
        print(f"Loading state dict from fused temp file...")
        state_dict = torch.load(temp_filepath, map_location=map_location)
        return state_dict
    finally:
        # Clean up temp file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
