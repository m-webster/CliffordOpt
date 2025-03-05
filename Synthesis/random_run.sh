#!/bin/bash -l
# /usr/local/bin/python3 Synthesis/random_run.py -f sym_small -m pytket -s 0 --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_small -m pytket -s 1 --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_small -m pytket -s 2 --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_small -m pytket -s 3 --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_small -m rustiq --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_small -m stim --mode sym

/usr/local/bin/python3 Synthesis/random_run.py -f sym_large -m pytket -s 0 --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_large -m pytket -s 1 --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_large -m pytket -s 2 --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_large -m pytket -s 3 --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_large -m rustiq --mode sym
/usr/local/bin/python3 Synthesis/random_run.py -f sym_large -m stim --mode sym