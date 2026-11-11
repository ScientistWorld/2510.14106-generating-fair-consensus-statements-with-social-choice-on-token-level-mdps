#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_crash --instances "$1" --outputs "$2"
