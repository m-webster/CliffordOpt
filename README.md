# CliffordOpt 
Python package for heuristic and optimal Clifford circuit synthesis.


## Installation
PyPI: `pip install cliffordopt`

Source code: Download this repository and run `pip install .`

## Overview
This software enables users to synthesize Clifford and CNOT circuits.
Users can choose to minimise either the two-qubit gate count or the circuit depth.
 

## Synthesizing a Circuit

There are three main functions for synthesizing a circuit depending on the type of circuit:

1. synth_GL(U,params): synthesize a CNOT circuit from its n x n binary parity matrix U. This script is set up for use in examples/test_GL.py - users can either enter the matrix as a string or generate a random invertible matrix;
2. synth_Sp(U,params): synthesize a general Clifford circuit from its 2n x 2n symplectic matrix U. This script is set up for use in examples/test_Sp.py - users can either enter the matrix as a string or generate a random invertible matrix;
3. synth_QC(qc,params): synthesize a general Clifford circuit from a string representing the circuit in QASM2 format. This script is set up for use in examples/test_QC.py - users can paste in a QASM2 string to synthesize a circuit.

Users can also synthesize encoding circuits for quantum error correction codes using the examples/test_QECC.py script.
We give examples of encoding circuits of the toric code, codetables.de codes and CSS codes.

## Input Parameters for the Algorithms

The input to the algorithms are a binary symplectic matrix *U* and a an object *params* which specifies parameters for the search criteria.

The main settings for the *params* are as follows:
- **mode**: "GL" for CNOT circuits (GL(n,2) matrices) - in this case, the circuit is synthesized using CNOT gates and may involve an initial qubit permutation.  "Sp" for general Clifford circuits (Sp(n,2) matrices) - in this case, the circuit is synthesized using 2-qubit transvections and may involve an initial qubit permutation followed by a set of single-qubit Clifford gates;
- **minDepth**: False to optimise for two-qubit gate count and True to optimise for circuit depth;
- **method**: the synthesis method. Valid options are  'greedy', 'astar', 'optimal', 'CNOT_greedy', 'CNOT_depth' ,'CNOT_gaussian','CNOT_Patel', 'CNOT_brug', 'volanto', 'stim', 'rustiq', 'qiskit', 'pytket', 'pyzx';
- **methodName**: for qiskit and pytket, there are a number of different pass types which can be selected. For qiskit, options are 'greedy' or 'ag'. For pytket, options are  'FullPeepholeOptimise', 'CliffordSimp','SynthesiseTket' and 'CliffordResynthesis';
- **file**: read circuits from source file "file".

Parameters used for calculating the heuristic function for greedy and astar algorithms only include:
- **hl**: if non-zero,  use log of colsums for the heuristic *h*" otherwise use the sum of the matrix for *h*;
- **ht**: if non-zero, include the transpose of the matrix when calculating *h*;
- **hi**: if non-zero, include the inverse of the matrix when calculating *h*;
- **hv**: greedy only - if non-zero, use vector heuristic  else scalar heuristic
- **wMax**: for greedy only - the max number of gates to apply without improving the heuristic before abandoning. If set to zero, never abandon;
- **hr**: Astar only -  the weighting of colsums when calculating the heuristic *h* - default is 3;
- **qMax**: Astar only - the maximum size of the priority queue. If the size is exceeded, entries with the highest heuristic are removed from the queue;

## New Algorithms

The following new algorithms have been implemented:

1. **Optimal Synthesis** - *csynth_greedy(U,params)*: for circuits on a small number of qubits, we have constructed databases of all equivalence classes of circuits up to permutations of the input and output qubits and, for general Clifford circuits, single-qubit Cliffords acting on the left and right. Users can use optimal synthesis for either minimal depth or minimal gate count on CNOT circuits on up to 7 qubits abd general Clifford circuits on up to 5 qubits.
2. **Astar Synthesis** - *synth_astar(U,params)*: the Astar algorithm gives good results for intermediate-size circuits too large for optimal synthesis.
The main parameter which may need adjustment is the hr parameter - modification of this parameter can affect processing time and the optimality of results. By default, the H_prod metric is used though this can be changed to H_sum by setting hl=0.
3. **Greedy Synthesis** - *csynth_opt(U,params)*: this is a fast algorithm which can be used for large circuit sizes. By default, the algorithm uses a vector heuristic, but a scalar heuristic can be used by setting the parameter hv=0. In this case, the H_prod scalar metric is used though this can be changed to H_sum by setting hl=0.

## Benchmark Comparison Algorithms
The new algorithms can be benchmarked against those of other approaches discussed in the literature.

The following algorithms for CNOT circuits are available and act on an nxn binary invertible matrix *U*:

1. method='CNOT_greedy' - CNOT_greedy(U): minimum CNOT-count synthesis from Low-overhead Magic State Circuits with Transversal CNOTs https://arxiv.org/abs/2501.10291

2. method='CNOT_depth' - CNOT_greedy_depth(U): minimum CNOT circuit depth synthesis from Low-overhead Magic State Circuits with Transversal CNOTs https://arxiv.org/abs/2501.10291 

3. method = 'CNOT_gaussian' - CNOT_GaussianElim(U): CNOT synthesis via Gaussian eliminiation.

4. method = 'CNOT_Patel' -  CNOT_Patel(U): PMH Algorithm from Optimal synthesis of linear reversible circuits, https://ieeexplore.ieee.org/abstract/document/10313691

5. method = 'CNOT_brug' - CNOTBrug(U): triangular matrix CNOT synthesis from Brugiere et al: Gaussian Elimination versus Greedy Methods for the Synthesis of Linear Reversible Circuits, https://doi.org/10.1145/3474226

The following are methods for general Clifford circuits and take as input either a 2n x 2n binary symplectic matrix *U* or a circuit *qc* as a Qiskit quantum circuit object:

1. method = 'volanto' - csynth_volanto(U): modified Volanto synthesis from  *Minimizing the Number of Two-qubit Gates in Clifford Circuits*, https://doi.org/10.1007/978-3-031-26553-2_7;

6. method = 'stim' - csynth_stim(U): Gidney, *Stim: a fast stabilizer circuit simulator*, https://doi.org/10.22331/q-2021-07-06-497;

2. method = 'rustiq' - csynth_rustiq(U): rustiq: Brugiere et a, *A graph-state based synthesis framework for Clifford isometries*, https://doi.org/10.22331/q-2025-01-14-1589;

4. method = 'qiskit' - csynth_qiskit(qc,methodName): AG synthesis or greedy synthesis from *Clifford Circuit Optimization with Templates and Symbolic Pauli Gates*, https://doi.org/10.22331/q-2021-11-16-580. methodName options are 'greedy' or 'ag';

1. method = 'pytket' -  csynth_tket(qc,methodName):  pytket: Sivarajah et al, *t|ket⟩: a retargetable compiler for NISQ devices*, https://dx.doi.org/10.1088/2058-9565/ab8e92. methodName options are  'FullPeepholeOptimise', 'CliffordSimp','SynthesiseTket' and 'CliffordResynthesis';

3. method = 'pyzx' - csynth_pyzx(qc): Kissinger, *PyZX: Large Scale Automated Diagrammatic Reasoning*, http://dx.doi.org/10.4204/EPTCS.318.14.

## Running Multiple Circuits

The examples/run_MatFile.py and examples/run_MatFileSens.py scripts run the synthesis algorithms on the circuits in text files saved in the examples/MatFiles folder. 

These scripts are usually run via the command line interface - for example python3 run_random.py -f GL2_7 --method astar --hr 4 --mode GL.

The parameters are as described above and for this example, we analyse the circuits saved in the file examples/random/GL2_7.txt using the astar method with hr=4 and in the GL mode.

The format of the input text file is a list of binary strings, one on each line, representing either binary invertible matrices (mode=GL) or symplectic matrices (mode=Sp).

Synthesis is run in parallel via the concurrent.futures.ProcessPoolExecutor function. By default, we set nWorkers=None which uses all available cores, though this can be varied by the user.

The results of the analysis are saved into the examples/results folder - the number of entangling gates, circuit depth, processing time and the synthesized circuit are saved on a single line for each circuit.

The  examples/run_MatFileSens.py is used to do sensitivity analysis for the Astar algorithm.
Users can set a range for the values of hr by setting the following parameters:
- r0: starting value for hr
- r1: final value for hr
- rStep: the amount to incremeent hr in each step

Similarly, the examples/run_QCFile.py and examples/run_QCFileSens.py scripts run the circuits saved in the examples/QCFiles folder.
These files are saved in CSV format and the circuits are in QASM2 format.

## Optimal Database Generation and Analysis

The examples/optDB.py script illustrates the generation and analysis of databases for optimal synthesis.

The function ResumeDBGen(n,mode,nWorkers,minDepth ) calculates an optimal database for circuits on n qubits. 
It can be run using multiple threads in parallel and the calculations can be re-started in the case of server timeouts.
The databases are saved in SQLite format in the opt folder of the package.
We have saved databased up to n=6 qubits for CNOT circuits and n=5 circuits for general Clifford circuits, but users can generate larger databases as requried using the ResumeDBGen script.

The ResumeDBGen function inputs are as follows:
- n: number of qubits in the circuit
- mode: either 'GL' for CNOT circuits or 'Sp' for general Clifford circuits
- nWorkers: number of threads to run simultaneously
- minDepth: True to calculate minimum depth database or false for minimum gate count circuits.

The analyseOptDB(n,mode,minDepth) function prints out the number of equivalence classes which require either d 2-qubit gates or depth d, depending on the minDepth setting.

The correlDB(mode,n,minDepth) function calculates the H_sum and H_prod metrics, then calculates the correlation coefficient and slope of the line of best fit with d.

## Citation 
Paper



Software
```
@misc{cliffordopt,
author = {Webster, Mark},
license = {GNU},
month = mar,
title = {{CliffordOpt}},
url = {https://github.com/m-webster/CliffordOpt},
version = {1.0.0},
year = {2025}
}
```