#!/bin/bash
# run file with correct python path! 
# $ PYTHONPATH=. ./datalake/run_entire_pipeline.sh
uv run datalake/raw/hypotactic/run_poems.py
uv run datalake/raw/hypotactic/run_prose.py
uv run datalake/raw/pedecerto/run_pedecerto.py

uv run datalake/clean/run_clean_pedecerto.py
uv run datalake/clean/run_clean_hypotactic.py

uv run datalake/enriched/run_poetry.py
