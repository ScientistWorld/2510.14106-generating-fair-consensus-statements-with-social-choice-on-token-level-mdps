#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_invalid_nan --instances "$1" --outputs "$2"
