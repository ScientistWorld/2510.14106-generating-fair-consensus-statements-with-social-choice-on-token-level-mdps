#!/bin/bash
set -e
cd /home/user
python -m hardening.variants shortcut_fake_fields --instances "$1" --outputs "$2"
