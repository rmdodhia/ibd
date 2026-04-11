#!/bin/bash
# Run the Streamlit labeling UI
cd "$(dirname "$0")"
source .venv/bin/activate
PYTHONPATH=. streamlit run labeler/app.py "$@"
