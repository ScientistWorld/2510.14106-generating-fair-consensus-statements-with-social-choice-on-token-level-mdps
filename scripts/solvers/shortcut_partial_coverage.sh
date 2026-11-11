#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_partial_coverage --instances "$1" --outputs "$2"
