import subprocess
import concurrent.futures
import os

def sync_single_run(run_dir):
    """Executes the wandb sync command for a specific directory."""
    try:
        print(f"Starting sync for: {run_dir}")
        # 'wandb sync' is a CLI command, so we call it via subprocess
        result = subprocess.run(
            ["wandb", "sync", run_dir],
            capture_output=True,
            text=True,
            check=True
        )
        return f"SUCCESS: {run_dir}\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"FAILED: {run_dir}\nError: {e.stderr}"

def sync_all_runs_parallel(run_dirs, max_workers=4):
    """Syncs multiple wandb directories in parallel."""
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map the sync function to the list of directories
        future_to_dir = {executor.submit(sync_single_run, d): d for d in run_dirs}
        
        for future in concurrent.futures.as_completed(future_to_dir):
            print(future.result())
            
    