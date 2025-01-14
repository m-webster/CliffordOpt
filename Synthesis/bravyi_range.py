import add_parent_dir
import numpy as np
from CliffordOps import *
from clifford_synthesis import *
from NHow import *
import concurrent.futures
import argparse
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="Source File for Circuits",type=str, default='n6_compiled')
    parser.add_argument("-m","--method", help="Synthesis method - options are voltano, greedy, astar, qiskit, stim, pytket",type=str, default='astar')
    parser.add_argument("--astarRange","-a", help="Astar, range of r1, r2 values",type=int, default=1)
    parser.add_argument("-r1", help="For astar only: r1 is the weighting of Rank1 matrices when calculating h",type=float, default=1)
    parser.add_argument("-r2", help="For astar only: r2 is the weighting of Rank2 matrices when calculating h",type=float, default=1)
    parser.add_argument("-q","--qMax", help="For astar only: max size of the priority queue.",type=int, default=10000)
    params = parser.parse_args()
    set_global_params(params)
    write_params(params)

    circuits = readBravyiFile(params.infile)
    # rVals = [(0.9 + (i+j) * 0.1,1+i * 0.1) for i in range(8) for j in range(10)]
    rVals = {(i/10,j/10) for i in range(9,21) for j in range(10,19) if (i-j) > 6 or (i-j) < -1}
    rVals = sorted(rVals,reverse=True)
    # print(rVals)
    paramList = []
    for (r1,r2) in rVals:
        p1 = copy.copy(params)
        p1.r1 = r1
        p1.r2 = r2
        paramList.append(p1)
    # i = 6
    # for p2 in paramList:
    #     bravyi_run(circuits,i,p2)
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        myrange = range(len(circuits))
        # myrange = [5,7,798,873]
        threadFuture = {executor.submit(bravyi_run,circuits,i,p2): (i+1) for p2 in paramList for i in myrange }
        for future in concurrent.futures.as_completed(threadFuture):
            i = threadFuture[future]
