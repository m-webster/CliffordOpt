from cliffordopt import *
import cProfile

## paste invertible matrix here
## Fig1 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
# CX-count quoted: 3; opt: 3
# mytext = '''1111
# 0111
# 0011
# 0001'''

## Fig2 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
# CX-count quoted: 10; opt: 8
# opt-depth: 5
mytext = '''10011
01101
01110
10110
11001'''

## Fig7 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
# CX-count quoted: 12; opt: 8; opt-depth 4
# mytext = '''110110
# 101110
# 000101
# 010111
# 011111
# 000110'''

## Fig7 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
##CX-count quoted: 8; opt: 8; opt-depth 4
# mytext = '''111001
# 011011
# 001100
# 101101
# 101111
# 001001'''

## Fig11 from A Cost Minimization Approach to Synthesis of Linear Reversible Circuits
## quoted: 59
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

## Example from Gaussian elimination vs greedy methods
# mytext = '''1000000000
# 1100000000
# 1110000000
# 1111000000
# 0100100000
# 1100010000
# 1110011000
# 1011100100
# 1000010110
# 0001001111'''


###############################################
## Global parameters - don't change!!
###############################################
params = paramObj()
params.mode = 'GL'

###############################################
## optimal, greedy and astar settings
###############################################

## choose a method
params.method = 'optimal'
params.method = 'astar'
params.method = 'greedy'

## optimise for depth or gate count
params.minDepth = False

## heuristic settings
params.hv = 1 ## vector
params.hi = 1 ## include inverse
params.ht = 1 ## include transpose
params.hl = 1 ## log of cols 1 or sums 0
params.hr = 3 # scaling factor for heuristic

## greedy: max number of gates to apply before abandoning 
## if set to zero, never abandon
params.wMax = 0

## astar: 
params.qMax = 10000 # max priority queue length 

###############################################
## method: choose from 'pytket','qiskit','volanto','greedy','astar','stim'
###############################################

## algorithms from CCZ paper
# params.method = 'CNOT_greedy'
# params.method = 'CNOT_depth'

## Existing CNOT Synthesis Algorithms
# params.method = 'CNOT_gaussian'
# params.method = 'CNOT_Patel'

## Existing Clifford Synthesis Algorithms
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
## Run Algorithm
###############################################

U = bin2ZMat(mytext)

## random  matrix
# n=7
# U = GLRand(np.random.default_rng(), n)
# print(ZMatPrint(U))

# cProfile.run(f'synth_GL(U,params)')

## Synthesize circuit
n,gateCount,depth,procTime,check,circ = synth_GL(U,params)

if check != "":
    print(f'Check: {check}')
print(f'Entangling Gate Count: {gateCount}')
print(f'Circuit Depth: {depth}')
print(f'Processing time: {procTime}')
print(circ)
