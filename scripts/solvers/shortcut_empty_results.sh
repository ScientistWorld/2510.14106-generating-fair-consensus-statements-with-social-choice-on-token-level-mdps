#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_empty_results --instances "$1" --outputs "$2"
