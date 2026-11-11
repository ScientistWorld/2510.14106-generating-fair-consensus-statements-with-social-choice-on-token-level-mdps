#!/bin/bash
set -e
cd /home/user
python -m hardening.variants better_colgen --instances "$1" --outputs "$2"
