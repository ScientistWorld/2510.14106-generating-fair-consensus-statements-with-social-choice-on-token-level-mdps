#!/bin/bash
# Reference solver (the paper's method): Nash-welfare lottery over complete statements
# (Algorithm 1), token chunking when the query budget cannot cover the tree (Section 4.3),
# induced stochastic token policy (Definition 2).
# Interface (all solvers): started with no arguments, speaks the oracle protocol on
# stdin/stdout (see data/core_synthetic/oracle_client.py).
set -e
cd /home/user
exec python -m method.solve
