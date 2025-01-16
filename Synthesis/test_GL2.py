import add_parent_dir
import numpy as np
from CliffordOps import *
from clifford_synthesis import *
from NHow import *

## paste invertible matrix here
## Fig1 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
# min: 3
mytext = '''1111
0111
0011
0001'''

## Fig2 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
# min: 10 - slow for astar
mytext = '''10011
01101
01110
10110
11001'''

## Fig7 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
# min: 12
mytext = '''110110
101110
000101
010111
011111
000110'''

## Fig7 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
## min: 8
# mytext = '''111001
# 011011
# 001100
# 101101
# 101111
# 001001'''

## Fig11 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
## min: 59
# mytext = '''1010100101010011
# 1110011111011010
# 1111101100110001
# 1000100011010000
# 1101000011011000
# 0101011100010011
# 0000000000000100
# 0100011011011010
# 1010101100010101
# 1101011000101110
# 1111100010000011
# 0101010101101110
# 1111010010100011
# 0001011100000011
# 1110110100001010
# 1001011100010100'''

U = bin2ZMat(mytext)

###############################################
## method: choose from 'pytket','qiskit','volanto','greedy','astar','stim'
###############################################
method = 'astar'

###############################################
## methodName blank, apart from:
###############################################
## Qiskit: 'greedy','ag'
## Pytket: 'FullPeepholeOptimise','CliffordSimp','SynthesiseTket','CliffordResynthesis'
methodName = "greedy"

###############################################
## For astar only
###############################################
r1,r2 = 1.68,1.48
qMax = 10000

r,t,c,check = synth_GL2(U,method,r1,r2,qMax,methodName)

if check != "":
    print(f'Check: {check}')
print(f'Entangling Gate Count: {r}')
print(f'Processing time: {t}')
print(c)