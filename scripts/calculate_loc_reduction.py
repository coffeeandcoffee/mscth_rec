import os

def count_loc(directory):
    total_loc = 0
    total_bytes = 0
    for root, dirs, files in os.walk(directory):
        # Exclude common environment/cache folders entirely to prevent counting external libraries
        dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'env', '.env', '__pycache__', 'site-packages', 'node_modules', '.git']]
        
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        total_loc += len(lines)
                        total_bytes += sum(len(line.encode('utf-8')) for line in lines)
                except Exception as e:
                    pass
    return total_loc, total_bytes

if __name__ == "__main__":
    # Define paths (relative or absolute)
    old_repo_path = '/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests'
    new_repo_path = '/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/eeg_tiktok_pipeline'
    
    print("Calculating Lines of Code (LOC) and bytes for old vs new codebase...")
    
    old_loc, old_bytes = count_loc(old_repo_path)
    new_loc, new_bytes = count_loc(new_repo_path)
    
    print(f"\nOriginal Codebase (including old scripts, notebooks, unused pipelines):")
    print(f"LOC: {old_loc:,}")
    print(f"Bytes: {old_bytes:,}")
    
    print(f"\nNew Consolidated Pipeline (eeg_tiktok_pipeline):")
    print(f"LOC: {new_loc:,}")
    print(f"Bytes: {new_bytes:,}")
    
    if old_loc > 0:
        discarded_loc_pct = (1 - (new_loc / old_loc)) * 100
        print(f"\nReduction / Superseded LOC: {discarded_loc_pct:.2f}%")
        print(f"(Representing a reduction from {old_loc:,} to {new_loc:,} total lines of code)")
