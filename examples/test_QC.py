from cliffordopt import *

## paste cicuit text in qasm2 format here

## Bravyi 6-qubit circuit_57_8
mytext = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
x q[2];
z q[0];
swap q[0], q[1];
h q[4];
cx q[4], q[2];
swap q[1], q[2];
swap q[2], q[3];
h q[3];
s q[3];
h q[3];
s q[3];
s q[5];
h q[5];
s q[5];
h q[5];
swap q[4], q[5];
cx q[4], q[5];
s q[4];
h q[4];
s q[4];
s q[5];
h q[5];
s q[5];
swap q[3], q[5];
cx q[3], q[5];
cx q[5], q[4];
h q[3];
s q[5];
h q[5];
s q[5];
h q[5];
swap q[2], q[3];
cx q[4], q[3];
cx q[2], q[3];
swap q[1], q[2];
cx q[1], q[2];
h q[1];
s q[2];
h q[2];
s q[2];
swap q[0], q[1];
cx q[4], q[1];
cx q[1], q[3];
h q[4];'''


## Bravyi 6-qubit circuit_60_13
# mytext = '''OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[6];
# z q[5];
# x q[3];
# x q[2];
# z q[2];
# x q[0];
# cx q[1], q[0];
# s q[2];
# s q[2];
# h q[2];
# s q[2];
# h q[4];
# cx q[4], q[2];
# swap q[1], q[2];
# s q[2];
# h q[2];
# s q[2];
# h q[4];
# s q[4];
# h q[4];
# s q[4];
# swap q[3], q[4];
# s q[4];
# h q[4];
# s q[4];
# h q[5];
# swap q[4], q[5];
# cx q[5], q[4];
# s q[5];
# h q[5];
# s q[5];
# h q[5];
# h q[4];
# swap q[3], q[4];
# cx q[5], q[4];
# cx q[4], q[3];
# h q[5];
# h q[4];
# s q[3];
# swap q[2], q[4];
# cx q[5], q[4];
# cx q[3], q[4];
# cx q[4], q[2];
# s q[3];
# h q[3];
# s q[3];
# h q[4];
# s q[2];
# swap q[1], q[3];
# cx q[3], q[4];
# cx q[3], q[1];
# s q[3];
# s q[1];
# swap q[0], q[3];
# cx q[3], q[5];
# cx q[3], q[4];
# s q[3];
# h q[3];
# s q[3];
# s q[4];
# s q[3];'''

## CX12
# mytext = '''OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[2];
# cx q[0], q[1];'''

## CZ2
# mytext = '''OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[2];
# cz q[0], q[1];'''

## SWAP12
# mytext = '''OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[2];
# swap q[0], q[1];'''

###############################################
## Global parameters - don't change!!
###############################################
params = paramObj()
params.mode = 'Sp'

###############################################
## optimal, greedy and astar settings
###############################################
params.method = 'optimal' # available up to n=5
params.method = 'astar'
params.method = 'greedy'

## optimise for depth or gate count
params.minDepth = False

## heuristic settings
params.hv = 1 ## vector
params.hi = 1 ## include inverse
params.ht = 1 ## include transpose
params.hl = 1 ## log of cols 1 or sums 0

## greedy: max number of gates to apply before abandoning 
## if set to zero, never abandon
params.wMax = 0

## astar: 
params.qMax = 10000 # max priority queue length 
params.hr = 3 # scaling factor for heuristic

###############################################
## Existing Clifford Synthesis Algorithms
###############################################

# params.method = 'volanto'
# params.method = 'rustiq'
# params.method = 'stim'
# params.method = 'pyzx'

## Qiskit: methodName in ['greedy','ag']
# params.method = 'qiskit'
# params.methodName = "greedy"

## Pytket: methodName in ['FullPeepholeOptimise','CliffordSimp','SynthesiseTket','CliffordResynthesis']
# params.method = 'pytket'
# params.methodName = "FullPeepholeOptimise"

###############################################
## Run Synthesis Algorithm
###############################################

n,gateCount,depth,procTime,check,circ = synth_QC(mytext,params)

if check != "":
    print(f'Check: {check}')
print(f'Entangling Gate Count: {gateCount}')
print(f'Circuit Depth: {depth}')
print(f'Processing time: {procTime}')
print(circ)
