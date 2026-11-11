#!/bin/bash
set -e
cd /home/user
python -m hardening.variants weaker_coarse_chunks --instances "$1" --outputs "$2"
