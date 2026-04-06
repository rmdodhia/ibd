#!/bin/bash
# Run the Streamlit labeling UI
cd "$(dirname "$0")"
PYTHONPATH=. streamlit run labeler/app.py "$@"
