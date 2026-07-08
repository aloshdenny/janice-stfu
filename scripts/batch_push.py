import subprocess
import os
import sys

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd[:10])}{' ...' if len(cmd) > 10 else ''}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"Error: {res.stderr.decode('utf-8')}")
        sys.exit(res.returncode)
    return res.stdout.decode('utf-8')

def main():
    # 1. Get list of all modified and untracked files
    status_out = run_cmd(['git', 'status', '-uall', '--porcelain'])
    
    files_to_push = []
    for line in status_out.splitlines():
        if not line.strip():
            continue
        # status is first 2 characters, file path starts at index 3
        status = line[:2]
        fp = line[3:].strip().strip('"').replace('\\\\', '\\')
        
        if os.path.exists(fp) and os.path.isfile(fp):
            files_to_push.append((fp, os.path.getsize(fp)))
            
    print(f"Found {len(files_to_push)} files to commit and push.")
    total_size = sum(sz for _, sz in files_to_push)
    print(f"Total size: {total_size / 1e9:.3f} GB")
    
    if not files_to_push:
        print("No files to push.")
        return

    # Group files into batches of at most 1.2 GB (1.2 * 10^9 bytes)
    MAX_BATCH_SIZE = 1.2 * 1000 * 1000 * 1000  # 1.2 GB
    
    batches = []
    current_batch = []
    current_batch_size = 0
    
    # Sort files to group them efficiently
    for fp, sz in sorted(files_to_push, key=lambda x: x[1], reverse=True):
        if sz > MAX_BATCH_SIZE:
            print(f"Warning: File {fp} is larger than 1.2 GB ({sz/1e6:.1f} MB)! Adding alone.")
            batches.append([(fp, sz)])
            continue
            
        if current_batch_size + sz > MAX_BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []
            current_batch_size = 0
            
        current_batch.append((fp, sz))
        current_batch_size += sz
        
    if current_batch:
        batches.append(current_batch)
        
    print(f"Created {len(batches)} batches.")
    for i, batch in enumerate(batches):
        batch_sz = sum(sz for _, sz in batch)
        print(f"  Batch {i+1}: {len(batch)} files, {batch_sz / 1e6:.1f} MB")
        
    # Execute batch uploads
    for i, batch in enumerate(batches):
        print(f"\n--- Uploading Batch {i+1}/{len(batches)} ({sum(sz for _, sz in batch)/1e6:.1f} MB) ---")
        
        # Stage files in batch (chunked to avoid command length limits)
        chunk_size = 100
        for j in range(0, len(batch), chunk_size):
            chunk_files = [fp for fp, _ in batch[j:j+chunk_size]]
            run_cmd(['git', 'add'] + chunk_files)
            
        # Commit batch
        run_cmd(['git', 'commit', '-m', f"Batch upload part {i+1} of {len(batches)}"])
        
        # Push batch
        run_cmd(['git', 'push', 'origin', 'main'])
        print(f"Successfully pushed Batch {i+1}!")
        
    print("\nAll batches successfully committed and pushed to main!")

if __name__ == '__main__':
    main()
