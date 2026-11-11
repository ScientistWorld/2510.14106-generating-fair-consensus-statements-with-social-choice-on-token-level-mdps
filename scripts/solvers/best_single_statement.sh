#!/bin/bash
set -e
cd /home/user
python -m hardening.variants best_single_statement --instances "$1" --outputs "$2"
