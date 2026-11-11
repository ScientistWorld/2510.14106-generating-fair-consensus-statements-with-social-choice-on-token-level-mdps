#!/bin/bash
set -e
cd /home/user
python -m hardening.variants weaker_uniform --instances "$1" --outputs "$2"
