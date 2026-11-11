#!/bin/bash
set -e
cd /home/user
python -m hardening.variants maxmin_in_core --instances "$1" --outputs "$2"
