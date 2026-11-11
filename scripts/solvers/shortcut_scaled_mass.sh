#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_scaled_mass --instances "$1" --outputs "$2"
