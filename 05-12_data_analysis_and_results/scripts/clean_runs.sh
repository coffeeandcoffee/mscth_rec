#!/bin/bash
# =============================================================================
# clean_runs.sh — Purge heavy intermediate data from runs
# =============================================================================
# Usage:
#   bash clean_runs.sh                     # Interactive menu
#   bash clean_runs.sh "run_20260526"      # Clean a specific run folder
# =============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUNS_DIR="$DIR/../runs"

if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Runs directory not found at $RUNS_DIR"
    exit 1
fi

clean_folder() {
    local run_folder="$1"
    if [ ! -d "$run_folder" ]; then
        echo "Error: Directory $run_folder does not exist."
        return
    fi
    
    local run_name=$(basename "$run_folder")
    echo -n "Cleaning up $run_name..."
    
    # Calculate size before (suppressing errors)
    local size_before=$(du -sm "$run_folder" 2>/dev/null | cut -f1)
    
    # Aggressively delete the heavy directories
    rm -rf "$run_folder/processed"
    rm -rf "$run_folder/windows"
    rm -rf "$run_folder/splits"
    rm -rf "$run_folder/features"
    rm -rf "$run_folder/features_notch"
    rm -rf "$run_folder/features_notch_artifact"
    rm -rf "$run_folder/features_notch_burst"
    rm -rf "$run_folder/features_notch_artifact_burst"
    rm -rf "$run_folder/results"
    
    # Calculate size after
    local size_after=$(du -sm "$run_folder" 2>/dev/null | cut -f1)
    
    local freed=$((size_before - size_after))
    echo " Freed ~${freed} MB"
    
    return $freed
}

echo "========================================================="
echo "  EEG PIPELINE - RUN FOLDER CLEANUP"
echo "========================================================="

# If an argument is provided, clean that specific folder
if [ -n "$1" ]; then
    target="$RUNS_DIR/$1"
    echo "Target: $1"
    read -p "Are you sure you want to clean this run? [y/N]: " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        clean_folder "$target"
        echo "Done!"
    else
        echo "Aborted."
    fi
    exit 0
fi

# Interactive Menu
echo "Select a run to clean (deletes heavy .pkl files, keeps parameters/vizes/csvs):"
echo ""

# Gather all runs
runs=()
for d in "$RUNS_DIR"/run_*; do
    if [ -d "$d" ]; then
        runs+=("$(basename "$d")")
    fi
done

if [ ${#runs[@]} -eq 0 ]; then
    echo "No runs found in $RUNS_DIR"
    exit 0
fi

# Add special options
options=("Clean ALL runs" "${runs[@]}" "Cancel")

PS3="Enter your choice (1-${#options[@]}): "
select opt in "${options[@]}"; do
    case "$REPLY" in
        1)
            echo ""
            read -p "WARNING: You are about to clean ALL ${#runs[@]} runs. Proceed? [y/N]: " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                total_freed=0
                for r in "${runs[@]}"; do
                    clean_folder "$RUNS_DIR/$r"
                    freed=$?
                    total_freed=$((total_freed + freed))
                done
                echo "========================================================="
                echo "Cleanup complete! Total disk space recovered: ~${total_freed} MB"
            else
                echo "Aborted."
            fi
            break
            ;;
        $(( ${#options[@]} )))
            echo "Cancelled."
            break
            ;;
        *)
            if [ "$REPLY" -gt 1 ] && [ "$REPLY" -lt "${#options[@]}" ]; then
                target_run="${options[$((REPLY-1))]}"
                echo ""
                read -p "Clean $target_run? [y/N]: " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    clean_folder "$RUNS_DIR/$target_run"
                    echo "Done!"
                else
                    echo "Aborted."
                fi
                break
            else
                echo "Invalid option. Try again."
            fi
            ;;
    esac
done
