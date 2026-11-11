#!/bin/bash
set -e
cd /home/user
python -m hardening.variants better_more_chunks --instances "$1" --outputs "$2"
