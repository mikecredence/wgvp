#!/bin/bash
# One command for a data folder (CPU stages + GPU stage when gvhmr.root/python are set in <root>/wgvp_config.json):
#   bash run_session.sh /path/to/data_root
# Uses the Python that runs this script's environment (pip install -r requirements.txt first). The GUI: python -m wgvp
set -e
cd "$(dirname "$0")"
python -m wgvp run "$@"
