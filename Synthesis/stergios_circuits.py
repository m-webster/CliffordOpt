import add_parent_dir
import numpy as np
from CliffordOps import *
from clifford_synthesis import *
from NHow import *


## paste cicuit text in qasm format here
mytext = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
h q[0];
s q[0];
y q[2];
h q[4];
s q[4];
h q[5];
s q[5];
y q[7];
h q[9];
s q[9];
cx q[0], q[5];
cx q[2], q[7];
cx q[4], q[9];
cx q[0], q[9];
cx q[2], q[5];
cx q[4], q[7];
cx q[0], q[7];
cx q[2], q[9];
cx q[4], q[5];
s q[0];
z q[0];
h q[0];
y q[2];
s q[4];
z q[4];
h q[4];
s q[5];
z q[5];
h q[5];
y q[7];
s q[9];
z q[9];
h q[9];'''

###############################################
## method: choose from 'pytket','qiskit','voltano','greedy','astar','stim'
###############################################
method = pytket

###############################################
## methodName blank, apart from:
###############################################
## Qiskit: 'greedy','ag'
## Pytket: 'FullPeepholeOptimise','CliffordSimp','SynthesiseTket','CliffordResynthesis'
methodName = "FullPeepholeOptimise"

qc = qiskit.QuantumCircuit.from_qasm_str(mytext)
S = qiskit.quantum_info.Clifford(qc)
U = ZMat(S.symplectic_matrix)

m,n = symShape(U)

## starting time
sT = currTime()

## PyTket
if method == 'pytket':
    opList = csynth_tket(mytext,methodName)

## Qiskit
elif method == 'qiskit':
    circ = csynth_qiskit(S,methodName)
    opList = qiskit2opList(circ)

## Voltano
elif method == 'voltano':
    vList, ix, CList = csynth_voltano(U)
    opList = trans2opList(vList,ix,CList)

## Greedy Algorithm
elif method == 'greedy':
    vList, ix, CList = csynth_greedy(U)
    opList = trans2opList(vList,ix,CList)

## Astar
elif method == 'astar':
    vList, ix, CList = csynth_astar(U,r1,r2,qMax)
    opList = trans2opList(vList,ix,CList)
    
## STIM
elif method == 'stim':
    opList = csynth_stim(U)

## if no method specified, just count gates in input circuit
else:
    opList = qiskit2opList(qc)

print('Run time',currTime()-sT)
print('Entangling Gate Count',entanglingGateCount(opList))
print(opList2str(opList))