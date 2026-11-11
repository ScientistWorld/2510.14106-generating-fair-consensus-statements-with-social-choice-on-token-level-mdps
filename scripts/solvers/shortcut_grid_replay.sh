#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_grid_replay --instances "$1" --outputs "$2"
