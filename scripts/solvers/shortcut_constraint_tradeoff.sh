#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_constraint_tradeoff --instances "$1" --outputs "$2"
