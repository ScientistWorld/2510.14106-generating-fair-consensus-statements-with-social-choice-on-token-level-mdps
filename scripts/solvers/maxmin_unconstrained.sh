#!/bin/bash
set -e
cd /home/user
python -m hardening.variants maxmin_unconstrained --instances "$1" --outputs "$2"
