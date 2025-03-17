from cliffordopt import *
import cProfile
import re

def ToricTableau(L):
    HX,HZ=toric2D(L)
    return CSS2Tableau(HX,HZ)

## repetition code
def repCode(r,closed=True):
    '''Generate classical repetition code on r bits.
    If closed, include one dependent row closing the loop.'''
    s = r if closed else r-1 

    SX = ZMatZeros((s,r))
    for i in range(s):
        SX[i,i] = 1
        SX[i,(i+1)%r] = 1
    return SX

## Symmetric Hypergraph Product Code
def SHPC(T):
    '''Make symmetric hypergraph product code from T.
    T can either be a string or np array.
    Returns SX, SZ.'''
    T = bin2ZMat(T)
    H = matMul(T.T, T,2)
    return HPC(H,H)

def HPC(A,B):
    '''Make hypergraph product code from clasical codes A, B
    A and B can either be a string or np array.
    Returns SX, SZ.'''
    A = bin2ZMat(A)
    B = bin2ZMat(B)
    ma,na = np.shape(A)
    mb,nb = np.shape(B)
    ## Generate SX
    C = np.kron(A,ZMatI(mb))
    D = np.kron(ZMatI(ma),B)
    SX = np.hstack([C,D])
    ## Generate SZ
    C = np.kron(ZMatI(na) ,B.T)
    D = np.kron(A.T,ZMatI(nb))
    SZ = np.hstack([C,D])
    return SX, SZ

## build 2D toric code from repetition code and SHPC constr
def toric2D(r):
    '''Generate distance r 2D toric code using SHCP construction.
    Returns SX, SZ.'''
    A = repCode(r,closed=False)
    return SHPC(A)

def str2bin(mytext):
    '''strip any characters which are not 0, 1 or newline, plus any leading/trailing whitespace'''
    return re.sub(r"[^01\n]+", "", mytext).strip()

###############################################
## Toric Code of distance d - uncomment below to run
###############################################
d = 4
U = ToricTableau(d)

###############################################
## CSS Code - uncomment below to run
###############################################
## [[7,1,3]] Steane code
HX = '''1001101
0101011
0010111'''
HZ = HX

## [[4,2,2]]  code
HX = '''1111'''
HZ = HX

HX =  bin2ZMat(str2bin(HX))
HZ =  bin2ZMat(str2bin(HZ))
U = CSS2Tableau(HX,HZ)

###############################################
## Codetables.de - uncomment  to run
###############################################
## [[5,1,3]] code
mytext = '''      [1 0 1 0 1|0 0 1 1 0]
      [0 0 1 1 0|1 0 0 1 1]
      [0 1 1 1 1|0 0 0 0 0]
      [0 0 0 0 0|0 1 1 1 1]'''

## [[8,3,3]] code
mytext = '''      [1 0 0 0 1 0 1 1|0 0 1 0 1 1 0 1]
      [0 0 0 1 0 1 1 1|1 0 1 0 0 1 1 0]
      [0 1 0 0 1 1 1 0|0 0 1 1 1 0 1 0]
      [0 0 0 1 0 1 1 1|0 1 0 1 1 0 0 1]
      [0 0 1 1 1 0 1 0|0 0 0 1 0 1 1 1]'''

mytext = str2bin(mytext)
S0 = bin2ZMat(mytext)
U = CodeTableau(S0)

###############################################
## Global parameters - don't change!!
###############################################
params = paramObj()
params.mode = 'Sp'

###############################################
## optimal, greedy and astar settings
###############################################

## choose a method
params.method = 'optimal'
params.method = 'astar'
# params.method = 'greedy'

## optimise for depth or gate count
params.minDepth = True

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

## random  matrix
# n=7
# U = GLRand(np.random.default_rng(), n)
# print(ZMatPrint(U))

# cProfile.run(f'synth_GL(U,params)')

## Synthesize circuit
n,gateCount,depth,procTime,check,circ = synth_Sp(U,params)

if check != "":
    print(f'Check: {check}')
print(f'Entangling Gate Count: {gateCount}')
print(f'Circuit Depth: {depth}')
print(f'Processing time: {procTime}')
print(circ)
