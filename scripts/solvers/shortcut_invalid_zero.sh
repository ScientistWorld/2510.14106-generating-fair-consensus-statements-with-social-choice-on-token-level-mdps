#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_invalid_zero --instances "$1" --outputs "$2"
