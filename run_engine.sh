#!/bin/bash

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$SCRIPT_DIR"

# activates python environment
# SET YOUR ENVIRONEMTN ACTIVATION COMMAND HERE
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate nlp

# run UCI engine script, saves logs if crashes
python -m src.uci_engine 2> debug_log.txt
