import add_parent_dir
import numpy as np
from CliffordOps import *
from clifford_synthesis import *
from NHow import *
import concurrent.futures
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="Source File for Matrices",type=str, default='GL2_small')
    parser.add_argument("-m","--method", help="Synthesis method - options are voltano, greedy, astar, qiskit, stim, pytket",type=str, default='greedy')
    parser.add_argument("-s","--submethod", help="Submethod for qiskit or pytket only. Qiskit: 'greedy'=0,'ag'=1. Pytket: 'FullPeepholeOptimise'=0,'CliffordSimp'=1,'SynthesiseTket'=2,'CliffordResynthesis'=3",type=int, default=0)
    parser.add_argument("-r1", help="For astar only: r1 is the weighting of Rank1 matrices when calculating h",type=float, default=1)
    parser.add_argument("-r2", help="For astar only: r2 is the weighting of Rank2 matrices when calculating h",type=float, default=1)
    parser.add_argument("-q","--qMax", help="For astar only: max size of the priority queue.",type=int, default=10000)
    parser.add_argument("--astarRange","-a", help="Astar, range of r1, r2 values",type=int, default=0)
    params = parser.parse_args()
    params.infile = f'Synthesis/random/{params.file}.txt'

    temp = params.file.split("_")
    params.t = temp[0]
  
    set_global_params(params)
    write_params(params)

    UList = readMatFile(params.infile)
    # random_run(UList,1,params)

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        myrange = range(len(UList))
        threadFuture = {executor.submit(random_run,UList,i,params): (i+1) for i in myrange}
        for future in concurrent.futures.as_completed(threadFuture):
            i = threadFuture[future]
