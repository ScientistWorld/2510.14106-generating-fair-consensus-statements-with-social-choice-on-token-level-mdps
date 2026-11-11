#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_easy_instances --instances "$1" --outputs "$2"
