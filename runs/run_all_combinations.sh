#!/bin/bash
# Run all optimization combinations automatically
# Runs each combination of optimization type (organic/oxide) and objective version (complex/simple/standard)
# with 1 iteration each

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"



cd "$PROJECT_DIR"

echo "=================================================================================="
echo "Running all optimization combinations"
echo "=================================================================================="
echo "Project directory: $PROJECT_DIR"
echo ""

# Define all combinations
OPT_TYPES=("organic")
OBJECTIVE_VERSIONS=("paper")
NOISE_LEVELS=(0.1 0.15)  # All noise levels to test
N_ITER=300  # Number of iterations for each combination

TOTAL_COMBINATIONS=$((${#OPT_TYPES[@]} * ${#OBJECTIVE_VERSIONS[@]} * ${#NOISE_LEVELS[@]}))
echo "Total combinations to run: $TOTAL_COMBINATIONS"
echo "  Optimization types: ${OPT_TYPES[@]}"
echo "  Objective versions: ${OBJECTIVE_VERSIONS[@]}"
echo "  Noise levels: ${NOISE_LEVELS[@]}"
echo "  Iterations per combination: $N_ITER"
echo "=================================================================================="
echo ""

# Function to run a single optimization
run_optimization() {
    local opt_type=$1
    local objective_version=$2
    local noise_level=$3
    
    python3 <<PYTHON_EOF
import sys
from pathlib import Path
sys.path.insert(0, '${PROJECT_DIR}')
from example_usage import run_organic_optimization, run_oxide_optimization

try:
    if '${opt_type}' == 'organic':
        optimizer = run_organic_optimization(objective_version='${objective_version}', n_iter=${N_ITER}, noise_level=${noise_level})
    elif '${opt_type}' == 'oxide':
        optimizer = run_oxide_optimization(objective_version='${objective_version}', n_iter=${N_ITER}, noise_level=${noise_level})
    else:
        print(f"Error: Unknown optimization type: ${opt_type}")
        sys.exit(1)
    
    print(f"SUCCESS: ${opt_type}_${objective_version}_noise${noise_level}")
    print(f"Output directory: {optimizer.output_dir}")
except Exception as e:
    print(f"FAILED: ${opt_type}_${objective_version}_noise${noise_level}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF
}

COMBINATION_NUM=0
SUCCESSFUL=0
FAILED=0
FAILED_LIST=()

for opt_type in "${OPT_TYPES[@]}"; do
    for objective_version in "${OBJECTIVE_VERSIONS[@]}"; do
        for noise_level in "${NOISE_LEVELS[@]}"; do
            COMBINATION_NUM=$((COMBINATION_NUM + 1))
            COMBINATION_NAME="${opt_type}_${objective_version}_noise${noise_level}"
            
            echo ""
            echo "=================================================================================="
            echo "Combination $COMBINATION_NUM/$TOTAL_COMBINATIONS: $COMBINATION_NAME"
            echo "=================================================================================="
            
            START_TIME=$(date +%s)
            
            if run_optimization "$opt_type" "$objective_version" "$noise_level"; then
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))
                SUCCESSFUL=$((SUCCESSFUL + 1))
                echo "✓ Successfully completed $COMBINATION_NAME in ${DURATION} seconds"
            else
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))
                FAILED=$((FAILED + 1))
                FAILED_LIST+=("$COMBINATION_NAME")
                echo "✗ Failed to complete $COMBINATION_NAME after ${DURATION} seconds"
            fi
        done
    done
done

# Summary
echo ""
echo "=================================================================================="
echo "SUMMARY"
echo "=================================================================================="
echo "Total combinations: $TOTAL_COMBINATIONS"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed combinations:"
    for failed_combo in "${FAILED_LIST[@]}"; do
        echo "  - $failed_combo"
    done
fi

echo "=================================================================================="
echo "All combinations completed!"
echo "=================================================================================="

exit $FAILED
