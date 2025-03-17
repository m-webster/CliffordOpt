from cliffordopt import *

## paste symplectic matrix here:

mystr = '''110111
001100
010010
111111
100100
011000'''

U = bin2ZMat(mystr)

## uncomment for random  matrix
# n=4
# U = symRand(np.random.default_rng(), n)
# print('isSymplectic',isSymplectic(U))
# print(ZMatPrint(U))



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

n,gateCount,depth,procTime,check,circ = synth_Sp(U,params)

if check != "":
    print(f'Check: {check}')
print(f'Entangling Gate Count: {gateCount}')
print(f'Circuit Depth: {depth}')
print(f'Processing time: {procTime}')
print(circ)
