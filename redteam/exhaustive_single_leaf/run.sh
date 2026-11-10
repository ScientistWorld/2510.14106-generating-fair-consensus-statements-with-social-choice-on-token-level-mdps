#!/bin/bash
set -euo pipefail
cd /home/user
python -m redteam.runner exhaustive_single_leaf
