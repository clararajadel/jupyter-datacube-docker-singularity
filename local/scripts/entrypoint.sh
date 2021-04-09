#!/bin/bash
cd /app/opt
source venv/bin/activate
pip install ipykernel -t /app/opt/venv/bin/
python3 -m ipykernel install --user --name=odc-kernel

jupyter-notebook --config=/app/conf/jupyter.py

# /app is because the environment in the Dockerfile will be named /app and singularity needs the entire path to build the image
