#!/bin/bash
set -e
cd /home/user
python -m method.solve --init-seed 11 --instances "$1" --outputs "$2"
