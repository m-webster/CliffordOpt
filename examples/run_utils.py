import argparse
import sys
import os
import csv
from cliffordopt import *

########################################################
## Helper functions to synthesise circuits from file
########################################################

def synthSave(C,i,params,circuitName=None):
    '''Synthesize circuit C in text form. Interpretation of C depends on params.mode - can be a circuit or binary string'''
    CName = "" if circuitName is None else f'\t{circuitName}'
    if params.mode == 'QC':
        n,gateCount,depth,procTime,check,circ = synth_QC(C,params)
    else:
        U = bin2ZMat(C)[0]
        n = int(np.round(np.sqrt(len(U))))
        U = np.reshape(U,(n,n))
        if params.mode == 'GL':
            n,gateCount,depth,procTime,check,circ = synth_GL(U,params)
        else:
            n,gateCount,depth,procTime,check,circ = synth_Sp(U,params)
    f = open(params.outfile,'a')
    if params.astarRange:
        f.write(f'{i+1}{CName}\t{n}\t{params.hr}\t{gateCount}\t{depth}\t{procTime}\t{check}\t{circ}\n')
    else:
        f.write(f'{i+1}{CName}\t{n}\t{gateCount}\t{depth}\t{procTime}\t{check}\t{circ}\n')
    f.close()
    ## return result + exec time + opList
    return (i,n,gateCount,depth,procTime,check,circ)

######################################################
## Helper Functions to Run Scenarios
######################################################

def defaultParser():
    '''parser for command line python scripts eg random_run, random_range, bravyi_run, bravyi_range'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="Source File for Matrices",type=str, default='GL_7')
    parser.add_argument("-m","--method", help="Synthesis method - options are volanto, greedy, astar, qiskit, stim, pytket",type=str, default='greedy')
    parser.add_argument("-s","--submethod", help="Submethod for qiskit or pytket only. Qiskit: 'greedy'=0,'ag'=1. Pytket: 'FullPeepholeOptimise'=0,'CliffordSimp'=1,'SynthesiseTket'=2,'CliffordResynthesis'=3",type=int, default=0)
    parser.add_argument("--mode", help="Type of Matrix",type=str, default='GL')  
    parser.add_argument("--minDepth", help="Run Minimum Depth Optimisation",type=int, default=0)  
    parser.add_argument("-wMax", help="For greedy only: wMax is the max number of iterations without improvement before abandoning. If set to zero, never abandon.",type=int, default=0)
    parser.add_argument("-hv", help="For greedy only: hv=1 means vector h, float otherwise",type=int, default=1) 
    parser.add_argument("-hr", help="For astar only: r is the weighting of colsums when calculating h",type=float, default=3)
    parser.add_argument("-hl", help="For greedy/astar only: if log=1 the use log of colsums calculating h",type=int, default=1)
    parser.add_argument("-ht", help="For greedy/astar only: if t=1 add transpose when calculating h",type=int, default=1)
    parser.add_argument("-hi", help="For greedy/astar only: if i=1 add inverse calculating h",type=int, default=1)
    parser.add_argument("-q","--qMax", help="For astar only: max size of the priority queue.",type=int, default=1000)
    parser.add_argument("--astarRange","-a", help="Astar, range of r values",type=int, default=0)
    return parser

def set_global_params(params):
    '''Process parameters - set name of output file'''
    mydate = time.strftime("%Y%m%d-%H%M%S")
    ## for pytket, qiskit set methodName
    if params.method == 'pytket':
        methods = ['FullPeepholeOptimise','CliffordSimp','SynthesiseTket','CliffordResynthesis','Combo']
        params.methodName = methods[params.submethod]
    elif params.method == 'qiskit':
        methods = ['greedy','ag']
        params.methodName = methods[params.submethod]
    else:
        params.methodName = ""
    ## for astar, record r1, r2, qmax
    if params.astarRange:
        myfile = f"{params.file}-{params.method}-hr{params.hr}-l{params.hl}-t{params.ht}-i{params.hi}-q{params.qMax}-{mydate}.txt"
    elif params.method in {'astar','CNOT_astar'}:
        myfile = f"{params.file}-{params.method}-hr{params.hr}-l{params.hl}-t{params.ht}-i{params.hi}-q{params.qMax}-{mydate}.txt"
    elif params.method in {'qiskit','pytket'}:
        myfile = f"{params.file}-{params.method}-{params.methodName}-{mydate}.txt"
    else:
        myfile = f"{params.file}-{params.method}-{mydate}.txt"

    cwd = os.getcwd()
    params.outfile = f'{cwd}/results/{myfile}'
    
def write_params(params):
    '''Write search parameters to file and std output'''
    temp = [printObj(params)]
    temp.append("#########################################")
    temp.append("")
    mytext = "\n".join(temp)
    if params.outfile is not None:
        f = open(params.outfile,'w')
        f.write(mytext)
        f.close()
    print(mytext)

def readMatFile(fileName):
    '''Read matrices from file - each line is a matrix represented as a binary vector. Skip comments started with # as first character'''
    f = open(fileName)
    temp = []
    for s in f.read().split('\n'):
        if len(s) > 0 and s[0] != "#":
            s = s.split("\t")
            temp.append(s[0])
    return temp

def readQCFile(myfile):
    '''read circuits from CSV file in Bravyi format - each circuit is a QASM string'''
    csv.field_size_limit(sys.maxsize)
    circuitList,circuitNames = [],[]
    with open(myfile) as csvfile:
        csvReader = csv.reader(csvfile, dialect='excel')
        c = 0
        for row in csvReader:
            ## skip header row
            if c > 0:
                ## name of the circuit is in the first column
                circuitNames.append(row[0])
                ## circuits are in the second column
                circuitList.append(row[1].replace('""','"'))
            c+=1
    return circuitList, circuitNames