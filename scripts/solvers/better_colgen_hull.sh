#!/bin/bash
set -e
cd /home/user
python -m hardening.variants_hull --instances "$1" --outputs "$2"
