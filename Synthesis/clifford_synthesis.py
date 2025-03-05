import add_parent_dir
from common import *
from NHow import *
from CliffordOps import *
import numpy as np
import argparse
import qiskit, qiskit.circuit, qiskit.qasm2
from mqt import qmap, qecc
import stim
import csv
import treap
import sys
import os
import pyzx
import pynauty as pn
from scipy import stats
# from sklearn import linear_model
## comment out for cluster - does not compile
import pytket, pytket.tableau, pytket.passes, pytket.qasm
import rustiq

#########################################################################################################
## Main Synthesis Algorithms
#########################################################################################################

def synth_GL(U,params):
    ## convert GL2 to Symplectic
    S = symCNOT(U)
    ## make a Qiskit quantum circuit and Clifford from saved text
    qc = sym2qc(S)
    return synth_main(qc,S,params)

def synth_QC(mytext,params):
    ## make a Qiskit quantum circuit and Clifford from saved text
    qc = qiskit.QuantumCircuit.from_qasm_str(mytext)
    S = qiskit.quantum_info.Clifford(qc)
    U = ZMat(S.symplectic_matrix)
    return synth_main(qc,U,params)

def synth_main(qc,U,params):
    MWalgs = ['optimal','volanto','greedy','astar','CNOT_optimal','CNOT_gaussian','CNOT_Patel','CNOT_greedy','CNOT_astar','CNOT_depth']
    U1 = U.copy()
    m,n = symShape(U)
    ## starting time
    sT = currTime()

    ############################################################
    ## Paper algorithms
    ############################################################

    ## Greedy Algorithm
    if params.method == 'greedy':
        opList, UC = csynth_greedy(U,params)
        opList = mat2SQC(UC) + opList

    ## Optimal Algorithm
    elif params.method == 'optimal':
        opList = CNOT_opt(U[:m,:n]) if params.mode == 'GL2' else csynth_opt(U) 

    ## Astar
    elif params.method == 'astar':
        opList, UC = synth_astar(U,params)
        opList = mat2SQC(UC) + opList

    ############################################################
    ## CNOT Circuit Methods
    ############################################################

    ## CNOT algorithms from CCZ paper
    elif params.method == 'CNOT_greedy':
        ix,CXList = CNOT_greedy(U[:m,:n])
        opList = CNOT2opList(ix,CXList)

    elif params.method == 'CNOT_depth':
        ix,CXList = CNOT_greedy_depth(U[:m,:n])
        opList = CNOT2opList(ix,CXList)

    ## Previous CNOT algorithms
    elif params.method == 'CNOT_gaussian':
        ix,CXList = CNOT_GaussianElim(U[:m,:n])
        opList = CNOT2opList(ix,CXList)

    elif params.method == 'CNOT_Patel':
        opList = CNOT_Patel(U[:m,:n])

    elif params.method == 'CNOT_brug':
        opList = CNOTBrug(U[:m,:n])

    # ############################################################
    # ## General Clifford Methods
    # ############################################################

    elif params.method == 'pytket':
        opList = csynth_tket(qc2qasm(qc),params.methodName)

    elif params.method == 'rustiq':
        opList = csynth_rustiq(U)

    elif params.method == 'pyzx':
        opList = csynth_pyzx(qc2qasm(qc))

    elif params.method == 'qiskit':
        circ = csynth_qiskit(qiskit.quantum_info.Clifford(qc),params.methodName)
        opList = qiskit2opList(circ)

    elif params.method == 'volanto':
        opList, UC = csynth_volanto(U)
        opList = mat2SQC(UC) + opList
        
    elif params.method == 'stim':
        opList = csynth_stim(U)
    
    ## if no method specified, just count gates in input circuit
    else:
        opList = qiskit2opList(qc)

    depth = len(opListLayers(opList))
    gateCount = entanglingGateCount(opList)
    t = currTime()-sT
    c = opList2str(opList,ch=" ")
    if params.method in MWalgs:
        check = symTest(U1,opList)
    else:
        check = ""
    return gateCount,depth,t,c,check

##########################################################
## Utilities for main Synth Algorithms
##########################################################

def sym2qc(U):
    '''Convert symplectic matrix U to a qiskit circuit object'''
    return qiskit.quantum_info.Clifford(U).to_circuit()

def qc2qasm(qc):
    '''convert qiskit circuit object to qasm 2 string'''
    return qiskit.qasm2.dumps(qc)

def bravyi_run(circuits,i,params):
    '''Run optimisation for circuits in Bravyi et al'''
    circuitName, mytext = circuits[i]
    r,t,c,check = synth_QC(mytext,params)
    ## write results to file
    f = open(params.outfile,'a')
    if params.astarRange:
        f.write(f'{i+1}\t{circuitName}\t{params.r}\t{params.hr}\t{params.hl}\t{params.ht}\t{params.hi}\t{t}\t{check}\t{c}\n')
    else:
        f.write(f'{i+1}\t{circuitName}\t{r}\t{t}\t{check}\t{c}\n')
    f.close()
    ## return result + exec time + opList
    return (i,circuitName,r,t,c)

def random_run(UList,i,params):
    '''Run optimisation for randomly generated circuits'''
    U = UList[i]
    # print('params.t',params.t)
    if params.mode == 'GL2':
        gateCount,depth,t,c,check = synth_GL(U,params)
        m,n = U.shape
    else:
        mytext =  qc2qasm(sym2qc(U))
        gateCount,depth,t,c,check = synth_QC(mytext,params)
        m,n = symShape(U)

    f = open(params.outfile,'a')
    if params.astarRange:
        f.write(f'{i+1}\t{n}\t{params.hr}\t{params.hl}\t{params.ht}\t{params.hi}\t{gateCount}\t{depth}\t{t}\t{check}\t{c}\n')
    else:
        f.write(f'{i+1}\t{n}\t{gateCount}\t{depth}\t{t}\t{check}\t{c}\n')
    f.close()
    ## return result + exec time + opList
    return (i,gateCount,depth,t,c)


######################################################
## Helper Functions to Run Scenarios
######################################################

class paramObj(object):
    pass

def defaultParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="Source File for Matrices",type=str, default='GL2_7')
    parser.add_argument("-m","--method", help="Synthesis method - options are volanto, greedy, astar, qiskit, stim, pytket",type=str, default='greedy')
    parser.add_argument("-s","--submethod", help="Submethod for qiskit or pytket only. Qiskit: 'greedy'=0,'ag'=1. Pytket: 'FullPeepholeOptimise'=0,'CliffordSimp'=1,'SynthesiseTket'=2,'CliffordResynthesis'=3",type=int, default=0)
    parser.add_argument("--mode", help="Type of Matrix",type=str, default='GL2')  
    parser.add_argument("--minDepth", help="Run Minimum Depth Optimisation",type=int, default=0)  
    parser.add_argument("-wMax", help="For greedy only: wMax is the max number of iterations without improvement before abandoning. If set to zero, never abandon.",type=int, default=0)
    parser.add_argument("-hv", help="For greedy only: hv=1 means vector h, float otherwise",type=int, default=1) 
    parser.add_argument("-hr", help="For astar only: r is the weighting of colsums when calculating h",type=float, default=1)
    parser.add_argument("-hl", help="For greedy/astar only: if log=1 the use log of colsums calculating h",type=int, default=1)
    parser.add_argument("-ht", help="For greedy/astar only: if t=1 add transpose when calculating h",type=int, default=1)
    parser.add_argument("-hi", help="For greedy/astar only: if i=1 add inverse calculating h",type=int, default=1)
    parser.add_argument("-q","--qMax", help="For astar only: max size of the priority queue.",type=int, default=10000)
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
    params.outfile = f'{cwd}/Synthesis/results/{myfile}'
    
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
    f = open(fileName)
    mytext = f.read()
    mytext = mytext.split('\n')
    temp = []
    for s in mytext:
        s = s.split("\t")
        if len(s[0]) > 0:
            A = bin2ZMat(s[0])[0]
            n = int(np.round(np.sqrt(len(A))))
            temp.append(np.reshape(A,(n,n)))
    return temp

def readMatOptFile(fileName,mode):
    f = open(fileName)
    mytext = f.read()
    mytext = mytext.split('\n')
    temp = []
    for s in mytext:
        if len(s) > 0:
            myrow = s.split('\t')
            n = int(myrow[1])
            n2 = n if mode=='GL2' else n*2
            d = int(myrow[2])
            A = np.reshape(bin2ZMat(myrow[0]),(n2,n2))
            temp.append((A,n,d))
    return temp

def readBravyiFile(myfile):
    '''read circuits from CSV file'''
    csv.field_size_limit(sys.maxsize)
    temp = []
    with open(myfile) as csvfile:
        csvReader = csv.reader(csvfile, dialect='excel')
        c = 0
        for row in csvReader:
            ## skip header row
            if c > 0:
                ## circuits are in the second column
                mytext = row[1].replace('""','"')
                ## name of the circuit is in the first column
                temp.append([row[0],mytext])
            c+=1
    return temp

#################################################################################
## opList Manipulations
#################################################################################

SQC_tostr = {'1001':'I', '0110':'H','1101':'S','1011':'HSH','1110':'HS','0111':'SH'}
SQC_fromstr = {v : np.reshape([int(i) for i in k],(2,2)) for k,v in SQC_tostr.items()}

def opList2sym(opList,n):
    return opListApply(opList,ZMatI(2*n),True)

def symTest(U,opList):
    m,n = symShape(U)
    U2 = opList2sym(opList,n)
    return binMatEq(U,U2)

def CNOT2opList(ix,opList):
    return [('QPerm',ix)] + [('CX',[i,j]) for (i,j) in opList]

def isSQC(opType):
    global SQC_fromstr
    return opType in SQC_fromstr

def applySQC(U,opType,qList):
    m,n = symShape(U)
    q = qList[0]
    A = str2SQC(opType)
    Ui = matMul(U[:,[q,q+n]],A,2)
    U[:,[q,q+n]] = Ui
    return U

def SQC2str(A):
    global SQC_tostr
    return SQC_tostr[ZMat2str(A.ravel())] 

def str2SQC(mystr):
    global SQC_fromstr
    return SQC_fromstr[mystr]

def CList2opList(UC):
    temp = []
    ## dict for single-qubit Cliffords
    for i in range(len(UC)):
        c =  SQC2str(UC[i])
        ## don't add single-qubit identity operators
        if c != 'I':
            temp.append((c,[i]))
    return temp

def opListLayers(opList):
    # opList = QPerm2Front(opList)
    layers = []
    for opType,qList in opList:
        if opType not in {'QPerm','SWAP'} and len(qList) > 1:
            L = len(layers)
            i = L
            qList = set(qList)
            while i > 0 and len(qList.intersection(layers[i-1])) == 0:
                i = i-1
            if i == L:
                layers.append(qList)
            else:
                layers[i].update(qList)
    return layers

def applyTv2(U,acbd,ij):
    '''Fast method for multiplying binary matrix U by 2-qubit transvection (acbd,ij)'''
    m,n = symShape(U)
    i,j = ij
    ## support of v
    ix = [i,j,i+n,j+n]
    ## support of Omega vT
    ixH = [i+n,j+n,i,j]
    ## non-zero of abcd
    nZ = bin2Set(acbd)
    ## calc U Omega vT - this is a col vector - non-zero entries are rows of U which anti-commute with v
    C = ZMatZeros(2*m)
    for k in nZ:
        C ^= U[:,ixH[k]]
    ## calc U + (U Omega vT)v - add v only where the row anti-commutes
    ## same as adding C only where v is non-zero
    temp = U.copy()
    for k in nZ:
        temp[:,ix[k]] ^= C
    return temp

def applyOp(U,opType,qList,update=False):
    m,n = symShape(U)
    if not update:
        U = U.copy()
    if opType == 'QPerm':
        ix = ZMat(qList)
        ix = vecJoin(ix,n + ix)
        U = U[:,ix]
    elif opType == "CX":
        (i,j) = qList
        U[:,j] ^= U[:,i]
        U[:,n+i] ^= U[:,n+j]
    elif isTv2(opType):
        U = applyTv2(U,opType,qList)
    # elif isTv1(opType):
    #     U = applyTv1(U,opType,qList)
    elif isSQC(opType):
        U = applySQC(U,opType,qList)
    return U

def opListApply(opList,A,update=False):
    if not update:
        A = A.copy()
    for (opType,qList) in opList:
        A = applyOp(A,opType,qList,True)
    return A

def mat2SQC(UC):
    UR2 = symR2(UC)
    ix = permMat2ix(UR2)
    ixR = ixRev(ix)
    ## extract list of single-qubit cliffords
    CList =  [Fmat(UC,i,ix[i]) for i in ixR]
    temp =  CList2opList(CList)
    ## check if we have the trivial permutation - if not append a QPerm operator
    if not isIdPerm(ixR):
        temp = [('QPerm', ixR)] + temp
    return temp

def UC2ixCList(UC):
    '''Convert binary matrix with exactly one Fij in each row/col of rank 2
    to a permutation and list of single-qubit Cliffords'''
    ## extract qubit permutation
    UR2 = symR2(UC)
    ix = permMat2ix(UR2)
    ## extract list of single-qubit cliffords
    CList =  [Fmat(UC,i,ix[i]) for i in range(len(ix))]
    return ix, CList

def trans2opList(vList,ix,UC):
    '''Convert output of MW optimisations to opList
    vList: list of 2-qubit transvections
    ix: qubit permutataion
    UC: list of single-qubit Cliffords'''
    n = len(ix)
    temp = CList2opList(UC)
    ## check if we have the trivial permutation - if not append a QPerm operator
    if not np.all(ix == np.arange(n)):
        # temp.append(('QPerm', ix))
        temp.append(('QPerm', ixRev(ix)))
    ## process 2-qubit transvections
    for acbd, ij in vList:
        temp.append((acbd,ij))
    return temp

def opListInv(opList):
    '''return inverse of opList'''
    temp = []
    for (opType,qList) in reversed(opList):      
        qList = ZMat(qList) 
        if opType == 'QPerm':
            qList = ixRev(qList)
        elif opType == 'HS':
            opType = 'SH'
        elif opType == 'SH':
            opType = 'HS'
        temp.append((opType,tuple(qList)))
    return temp

def opListT(opList):
    '''return transpose of opList'''
    temp = []
    for (opType,qList) in reversed(opList):   
        qList = ZMat(qList) 
        if opType == 'CX':
            qList = tuple(reversed(qList))
        elif isTv2(opType):
            opType = TvTransp(opType)    
        elif opType == 'QPerm':
            qList = ixRev(qList)
        elif opType == 'S':
            opType = 'HSH'
        elif opType == 'HSH':
            opType = 'S'
        temp.append((opType,tuple(qList)))
    return temp

def opListTInv(opList):
    '''return Transpose inverse of opList'''
    temp = []
    for (opType,qList) in (opList):       
        if opType == 'CX':
            qList = tuple(reversed(qList))
        elif isTv2(opType):
            opType = TvTransp(opType)
        temp.append((opType,tuple(qList)))
    return temp

def entanglingGateCount(opList):
    '''count number of entangling gates in list of operataors opList'''
    c = 0
    for opName,qList in opList:

        if isEntangling(opName,qList):
            c += 1
    return c

def isEntangling(opName, qList):
    ## any gate actingh on more than one qubit which is not a SWAP or QPerm
    return len(qList) > 1 and opName != 'SWAP' and opName != "QPerm"

def Pauli2bin(mystr):
    '''convert Pauli string to tupe representing x and z components'''
    pauliX = {'I':0,'X':1,'Y':1,'Z':0}
    pauliZ = {'I':0,'X':0,'Y':1,'Z':1}
    return tuple([pauliX[a] for a in mystr] + [pauliZ[a] for a in mystr])

def str2opList(mystr):
    '''convert string to opList'''
    mystr = mystr.split(" ")
    temp = []
    for myOp in mystr:
        if len(myOp) > 3:
            opType,qList = myOp.split(":")
            if opType[0].upper() == 'T':
                opType = Pauli2bin(opType[1:])
            else:
                s = set(c for c in opType)
                if s.intersection({"0","1"}) == s:
                    opType = tuple([int(c) for c in opType])
            qList = [int(c) for c in qList.split(",")]
            temp.append((opType,qList))
    return temp

def opList2str(opList,ch="\n"):
    '''convert oplist to string rep'''
    pauli_list = ['I','X','Z','Y']
    temp = []
    for opName,qList in opList:
        if isTv2(opName):
            opName = ZMat(opName)
            xz = opName[:2] + 2 * opName[2:]
            P = pauli_list[xz[0]] + pauli_list[xz[1]]
            opName = f't{P}'
        elif typeName(opName) in ('tuple','ndarray'):
            opName = ZMat2str(opName)
        opName = opName.replace(" ","")
        qStr = ",".join([str(q) for q in qList])
        temp.append(f'{opName}:{qStr}')
    return ch.join(temp)

def isTv2(opType):
    '''check if opType is 2-qubit transvection'''
    return (typeName(opType) == 'tuple') and (len(opType) == 4)

def TvTransp(opType):
    '''Transpose of transvection  swap X and Z components'''
    m = len(opType)//2
    return opType[m:] + opType[:m]

def QPerm2Front(opList):
    '''move qubit permtuations to front of opList'''
    temp = []
    ixC = None
    for opType,qList in reversed(opList):
        qList = ZMat(qList)
        if opType == 'QPerm':
            ## update permutation
            ixC = qList if ixC is None else qList[ixC]
            ixR = ixRev(ZMat(ixC))
        else:
            ## update other operator types
            if ixC is not None:
                qList = [ixR[i] for i in qList]
            temp.append((opType,qList))
    if ixC is not None and not isIdPerm(ixC):
        temp.append(('QPerm',ixC))
    temp.reverse()
    return temp

def SQC2front(opList):
    '''move single-qubit Cliffords to front of opList'''
    CList = dict()
    temp = []
    for opType,qList in reversed(opList):
        if isTv2(opType):
            opType = ZMat(opType)
            for i in range(2):
                q = qList[i]
                if q in CList:
                    opType[i],opType[i+2] = matMul([opType[i],opType[i+2]],CList[q],2)[0]
            temp.append ((tuple(opType), qList))
        elif isSQC(opType):
            q = qList[0]
            A = str2SQC(opType)
            if q not in CList:
                CList[q] = A
            else:
                CList[q] = matMul(A,CList[q],2)
        elif opType == 'QPerm':
            # ixR = ixRev(qList)
            CList = {qList[q]: CList[q] for q in CList.keys()}
            temp.append ((opType, qList))
        else:
            temp.append ((opType, qList))
    for i in sorted(CList.keys(),reverse=True):
        opType = SQC2str(CList[i])
        if opType != 'I':
            temp.append((opType,[i]))
    temp.reverse()
    return temp

def isIdPerm(ix):
    '''return true if ix is an identity permutation'''
    return nonDecreasing(ix)

def sym2tuple(U,mode):
    '''flatten U to a tuple - in the case of GL2, just take U_XX component'''
    if mode == 'GL2':
        m,n = symShape(U)
        U = U[:m,:n]
    return tuple(np.reshape(U,-1))

def getOpListRec(A,visited):
    '''Recursive method to get opList from tree structure'''
    p,op = visited[A][0],visited[A][1]
    if p < 0:
        return []
    parentOps = getOpListRec(p,visited)
    opType,qList,Atrans,Ainv = op
    if Ainv:
        parentOps = opListInv(parentOps)
    if Atrans:
        parentOps = opListT(parentOps)
    return parentOps + [(opType,qList)]

def sym2components(U):
    '''Split Symplectic matrix U into components XX,XZ,ZX,ZZ for use in stim, qiskit, pytket'''
    n = len(U)//2
    xx=np.array(U[:n,:n],dtype=bool)
    xz=np.array(U[:n,n:],dtype=bool)
    zx=np.array(U[n:,:n],dtype=bool)
    zz=np.array(U[n:,n:],dtype=bool)
    return xx,xz,zx,zz

######################################################
## Random Clifford Generation
######################################################

def symRand(rng,n):
    '''generate random Clifford operator on n qubits'''
    stabs, destabs = symRandVec(rng, n)
    return vec2sym(stabs,destabs)

def vec2sym(stabs,destabs):
    '''convert stabs/destabs to symplectic matrix'''
    n = len(stabs)
    opList = vec2Tv(stabs,destabs)
    return opList2sym(opList,n)

def symRandVec(rng, r):
    '''generate a random Clifford on r qubits
    Output: stabs - a series of random Paulis on r, r-1, ... 1 qubits
    destabs - a series of Paulis which anticommute with the stabs'''
    stabs = []
    destabs = []
    for i in range(r):
        n = r-i
        done = False
        while not done:
            x = ZMat(rng.integers(2,size=2*n) )
            done = np.sum(x) > 0
        stabs.append(x)
        z = ZMat(rng.integers(2,size=2*n)) 
        c = PauliComm(x,ZMat([z])) 
        if c[0] == 0:
            ## find min j such that x[j] = 1
            j = bin2Set(x)[0]
            ## flip bit j of z, but in opposite component
            z[(n+j) % (2 * n)] ^= 1
        destabs.append(z)
    return stabs,destabs   

def vec2Tv(stabs,destabs):
    '''convert stabs and detabs to a permutation, series of single-qubit Cliffords and 2-qubit transvections'''
    n = len(stabs)
    ix = np.arange(n)
    CList = []
    opList = []
    for i in range(n):
        U = ZMatZeros((2,2*n))
        U[0,i:n] = stabs[i][:n-i]
        U[0,n+i:] = stabs[i][n-i:]
        U[1,i:n] = destabs[i][:n-i]
        U[1,n+i:] = destabs[i][n-i:]        
        vListi,UCi = csynth_volanto(U)
        opList.extend(opListInv(vListi))
        R2 = symR2(UCi)
        j = bin2Set(R2)[0]
        if j > i:
            ix[[i,j]] = ix[[j,i]]
        CList.append(Fmat(UCi,0,j))
    temp = []
    if not isIdPerm(ix):
        temp.append(('QPerm',ix))
    temp.extend(CList2opList(CList))
    temp.extend(opList)
    return temp

def GL2Rand(rng,n):
    '''generate random nxn invertible matrix'''
    xList = GL2RandVec(rng,n)
    return vec2GL2(xList)

########################################################
## nauty automorphisms
########################################################

def binMat2AdjDict(U):
    '''Transform binary matrix U to an adjacency matrix where rows are vertices of one colour and cols vertices of 2nd colour'''
    m,n = U.shape
    iList,jList = np.nonzero(U)
    if m > 0:
        jList+=m
    temp = {i:[] for i in range(m)}
    for (i,j) in zip(iList,jList):
        temp[i].append(j)
    return temp

mxnDict = dict()

def cliffAdjmxn(m,n):
    '''Create edges on rows/cols to restrict to single-qubit Cliffords and qubit swaps'''
    # save to global variable to reduce processing time
    global mxnDict
    if (m,n) not in mxnDict:
        ## rows
        temp = cliffAdj(m)
        ## add cols
        temp = dictListUpate(temp,cliffAdj(n,m)) 
        ## save to global variable
        mxnDict[(m,n)] = temp
    return mxnDict[(m,n)]

def cliffAdj(n,m=0):
    '''Create edges on rows/cols to restrict to single-qubit Cliffords and qubit swaps'''
    ## 3 times to reflect X, Z and X+Z components
    n3 = 3*n
    ## m is an offset - for creating the col edges
    m3 = 3*m
    return {i+m3:[(((i + j * n) %  n3) + m3) for j in range(1,3)] for i in range(n3)}

def dictListUpate(D1,D2):
    '''merge dictionaries D1,D2 of type k:list'''
    for k,v in D2.items():
        if k not in D1:
            D1[k] = v
        else:
            D1[k].extend(v)
    return D1

def sym2Graph(U):
    '''Create a graph from a symplectic matrix U'''
    m,n = symShape(U)
    ## create U_3n matrix with X, Z and X+Z components for rows and cols
    U3 = ZMatZeros((m*3,n*3))
    U3[:2*m,:2*n] = U
    U3[:2*m,2*n:] = U[:,:n] ^ U[:,n:]
    U3[2*m:,:] = U3[:m,:] ^ U3[m:2*m,:]
    temp = binMat2AdjDict(U3)
    ## add edges to restrict to single-qubit Cliffords and SWAP for rows/cols
    temp = dictListUpate(temp,cliffAdjmxn(m,n))
    return  temp

def GL2Canonize(U,lab=False):
    '''Canonize GL2 matrix U - default is to return certificate for equiv class. lab=True for canonical labels'''
    m,n = U.shape
    ADict = binMat2AdjDict(U)
    colouring = [set(range(m)),set(range(m,m+n))]
    G = pn.Graph(m + n,False,ADict,colouring)
    return ZMat(pn.canon_label(G)) if lab else pn.certificate(G)

def symCanonize(U,lab=False):
    '''Canonize symplectic matrix U - default is to return certificate for equiv class. lab=True for canonical labels'''
    m,n = symShape(U)
    ADict = sym2Graph(U)
    colouring = [set(range(3 * m)),set(range(3 * m, 3 * (m+n)))]
    G = pn.Graph(3 * (m + n),False,ADict,colouring)
    return ZMat(pn.canon_label(G)) if lab else pn.certificate(G)

#########################################################################################################
## Optimal Synthesis 
#########################################################################################################

## TODO: update to use SQLite database
def CNOT_opt(A):
    m,n = A.shape
    certDB,infoDB = readDB('GL2',n)
    Ainv = binMatInv(A)
    AOpts = matOpts(A,Ainv)
    CertOpts = [[GL2Canonize(B) for B in myrow] for myrow in AOpts]
    for Bcert, i in certDB.items():
        for trans in range(len(CertOpts)):
            for inv in range(len(CertOpts[0])):
                if CertOpts[trans][inv] == Bcert:
                    opList = getOpListRec(i,infoDB)
                    B = opList2sym(opList,n)[:n,:n]
                    return DB2GL2(AOpts[trans][inv],B,opList,trans,inv)
    return None

def DB2GL2(A,B,opList,trans,inv):
    # print(trans,inv)
    m,n = B.shape
    # opList = str2opList(opList)
    ixA = GL2Canonize(A,True)
    ixB = GL2Canonize(B,True)
    ixLR = ixA[ixRev(ixB)]
    perms = []
    for i in range(2):
        ix = ixLR[n*i:n*(i+1)] - m*i
        perms.append(ix)
    opList = [('QPerm',(perms[0]))] + opList + [('QPerm',ixRev(perms[1]))]
    if trans:
        opList = opListT(opList)
    if inv:
        opList = opListInv(opList)
    opList = QPerm2Front(opList)
    return opList

def csynth_opt(A):
    m,n = symShape(A)
    Ainv = symInverse(A)
    AOpts = matOpts(A,Ainv)
    CertOpts = [[symCanonize(B) for B in myrow] for myrow in AOpts]
    certDB,infoDB = readDB('sym',n)
    for Bcert, i in certDB.items():
        for trans in range(len(CertOpts)):
            for inv in range(len(CertOpts[0])):
                if CertOpts[trans][inv] == Bcert:
                    opList = getOpListRec(i,infoDB)
                    B = opList2sym(opList,n)
                    return DB2sym(AOpts[trans][inv],B,opList,trans,inv)
    return None

def DB2sym(A,B,opList,trans,inv):
    m,n = symShape(B)
    # opList = str2opList(opList)
    ixA = symCanonize(A,True)
    ixB = symCanonize(B,True)
    ixLR = ixA[ixRev(ixB)]
    E = EMat(n)
    EInv = EInvMat(n)
    perms,cliffs = [],[]
    for i in range(2):
        ix = ixLR[3*n*i:3*n*(i+1)] - 3*m*i
        UC = matMul(E, EInv[ix],2)[:2*n,:2*n]
        if i == 0:
            UC = UC.T
        ix,CList = UC2ixCList(UC)
        perms.append(ixRev(ix))
        cliffs.append(CList)
    opList = [(SQC2str(cliffs[0][i]),[i]) for i in range(n)] + [('QPerm',perms[0])] + opList + [(SQC2str(cliffs[1][i]),[i]) for i in range(n)] + [('QPerm',perms[1])]
    if trans:
        opList = opListT(opList)
    if inv:
        opList = opListInv(opList)
    opList = SQC2front(opList)
    opList = QPerm2Front(opList)
    return opList
            
def EMat(n):
    '''for conversion of permutation matrices to symplectic matrices'''
    ## (0,x,z) => (x,z,x+z) 
    E = np.array([[1,0,1],[0,1,1],[1,1,1]],dtype=int)
    In = np.eye(n,dtype=int)
    return np.kron(E,In)
    
def EInvMat(n):
    '''for conversion of permutation matrices to symplectic matrices'''
    ## (x,z,x+z) => (x,z,0) 
    E = np.array([[0,1,1],[1,0,1],[1,1,1]],dtype=int)
    In = np.eye(n,dtype=int)
    return np.kron(E,In)

def matOpts(A,Ainv):
    '''generate set of matrices - transpose, inverse. Only add if it is different to A'''
    temp = [A]
    if not binMatEq(A,Ainv):
        temp.append(Ainv)
    if binMatEq(A,A.T):
        return [temp]
    else:
        return [temp,[B.T for B in temp]]

def findMat(A,Ainv,visited,mode):
    '''serach for matrix in database'''
    m,n = symShape(A)
    SM = matOpts(A,Ainv)
    certMax, ti = None, None
    for trans in range(len(SM)):
        for inv in range(len(SM[0])):
            B = SM[trans][inv]
            BCert = GL2Canonize(B[:m,:n]) if mode=='GL2' else symCanonize(B)
            if certMax is None or certMax < BCert:
                certMax = BCert
                ti = (trans,inv)
            if BCert in visited:
                return True, BCert, (inv, trans)
    return False, certMax, ti


DBList = dict()

def readDB(mode,n):
    global DBList
    if (mode,n) in DBList:
        print('found DB',mode,n)
        return DBList[(mode,n) ]
    myfile = f'Synthesis/optimal/{mode}-DB-{n}.txt'
    certDB = dict()
    currId = 0
    infoDB = []
    with open(myfile,'r') as f:
        mytext = f.readline()
        while len(mytext) > 3:
            mytext = mytext.split('\t')
            Aid = int(mytext[0])
            res = str2opList(mytext[1])
            (opType, qList) = res[0] if len(res) == 1 else ("",[])
            ti = mytext[2]
            Atrans,Ainv = int(ti[0]),int(ti[1])
            Bop = (opType,qList,Atrans,Ainv)
            infoDB.append((Aid,Bop))
            opList = getOpListRec(currId,infoDB)
            B = opList2sym(opList,n)
            BCert = symCanonize(B) if mode=='sym' else GL2Canonize(B[:n,:n])
            certDB[BCert] = currId
            currId += 1
            mytext = f.readline()
    DBList[(mode,n)] = (certDB,infoDB)
    return certDB,infoDB
    
def GL2yVals(U):
    m,n = symShape(U)
    yVals = []
    B = U[:n,:n]
    Binv = U[n:,n:]
    R = matColSum(B.T)
    C = matColSum(B)
    RI = matColSum(Binv.T)
    CI = matColSum(Binv)
    ## sum + I
    yVals.append((matSum(B) - n)/n)
    yVals.append((matSum(B) + matSum(Binv) - 2*n)/2/n)
    ## prod + T, I, IT
    yVals.append(np.sum(np.log(R))/n)
    yVals.append(np.sum(np.log(R) + np.log(C))/2/n)
    yVals.append(np.sum(np.log(R) + np.log(RI))/2/n)
    yVals.append(np.sum(np.log(R) + np.log(RI)+ np.log(C) + np.log(CI))/4/n)
    
    # yVals.append(np.sum(R*np.log(R) + RI*np.log(RI)+C*np.log(C) + CI*np.log(CI))/4/n)
    return yVals

def SPyVals(U):
    m,n = symShape(U)
    yVals = []
    B = symR0(U) ^ 1
    R = matColSum(B.T)
    C = matColSum(B)
    RU = matColSum(U.T)
    CU = matColSum(U)
    ## sum
    yVals.append((matSum(B)-n)/n)

    ## prod
    yVals.append(np.sum(np.log(R) )/n)
    yVals.append(np.sum(np.log(R) + np.log(C) )/2/n)

    ## symplectic matrix
    yVals.append((matSum(U)-2*n)/2/n)
    yVals.append(np.sum(np.log(RU))/2/n)
    yVals.append(np.sum(np.log(RU) + np.log(CU) )/4/n)
    # yVals.append(np.sum(R*np.log(R) +C*np.sum(np.log(C)))/2/n)

    R = matColSum(U.T)
    C = matColSum(U)
    yVals.append(matSum(U)/n/2 - 1)
    yVals.append(np.sum(np.log(R) + np.log(C) )/4/n)
    yVals.append(np.sum(R*np.log(R) +C*np.sum(np.log(C)))/4/n)
    return yVals

def correlDB(mode,n):
    myfile = f'Synthesis/optimal/{mode}-DB-{n}.txt'
    print(f'Linear Regresssion {mode}({n})')
    currId = 0
    infoDB = []
    dList,yVals = [],[]
    yLabs = ['s','sI','p','pT','pI','pIT'] if mode=='GL2' else ['sR','pR','pRT','sU','pU','pUT'] 
    with open(myfile,'r') as f:
        mytext = f.readline()
        while len(mytext) > 3:
            mytext = mytext.split('\t')
            Aid = int(mytext[0])
            res = str2opList(mytext[1])
            (opType, qList) = res[0] if len(res) == 1 else ("",[])
            ti = mytext[2]
            Atrans,Ainv = int(ti[0]),int(ti[1])
            Bop = (opType,qList,Atrans,Ainv)
            infoDB.append((Aid,Bop))
            opList = getOpListRec(currId,infoDB)
            dList.append(len(opList))
            U = opList2sym(opList,n)
            if mode=='GL2':
                yVals.append(GL2yVals(U)) 
            else:
                yVals.append(SPyVals(U)) 
            currId += 1
            mytext = f.readline()
    print("\t".join(['Type','r','slope','intercept']))
    for i in range(len(yLabs)):
        slope, intercept, r, p, std_err = stats.linregress([y[i] for y in yVals],dList)
        print(f'{yLabs[i]}\t{r}\t{slope}\t{intercept}')
    return 1

def analyseFile(n,mode):
    myfile = f'Synthesis/optimal/{mode}-DB-{n}.txt'
    dCounts = ZMatZeros(20)
    with open(myfile,'r') as f:
        myrow = f.readline()
        while len(myrow) > 0:
            myrow = myrow.split("\t")
            dCounts[int(myrow[-1])] += 1
            myrow = f.readline()
    return dCounts

def extractRandom(n,mode,target=10):
    # myfile = f'Synthesis/optimal/{mode}-DB-{n}.txt'
    certDB,infoDB = readDB(mode,n)
    N = len(infoDB)
    dDict = [0] * N
    d = 0
    dix = []
    dCount = 1
    for i in range(N):
        Aid = infoDB[i][0]
        if Aid >= 0:
            dCurr = dDict[Aid] + 1
            dDict[i] = dCurr
            if dCurr > d:
                dix.append(dCount)
                dCount = 1
                d = dCurr
            else:
                dCount += 1
    dix.append(dCount)
    AList = []
    dList = []
    curr = 0
    for d in range(len(dix)):
        if dix[d] <= target:
            AList.extend(list(range(curr,curr+dix[d])))
            dList.extend([d] * (dix[d]))
        else:
            ix = np.random.choice(range(dix[d]),size=target)
            for i in ix:
                AList.append(curr + i)
                dList.append(d)
        curr += dix[d]
    temp = []
    for Aid in AList:
        opList = getOpListRec(Aid,infoDB)
        A = opList2sym(opList,n)
        if mode == 'GL2':
            A = applyRandPerm(A[:n,:n])
        else:
            A = applyRandPermSQC(A)
        temp.append(ZMat2str(A.ravel()))
    return temp,dList

def applyRandPerm(A):
    m,n = A.shape
    ixL = np.random.permutation(n)
    ixR = np.random.permutation(n)
    return A[ixL][:,ixR]

def applyRandPermSQC(A):
    cliff_list = ['1001','0110','1101','1011','1110','0111']
    cliff_list = [np.reshape(bin2ZMat(C),(2,2)) for C in cliff_list]
    m,n = symShape(A)
    for k in range(2):
        A = A.T
        ix = ZMat(np.random.permutation(n))
        A = A[vecJoin(ix,n+ix)]
        CList = np.random.randint(6,size=n)
        CList = symKron([cliff_list[C] for C in CList])
        A = matMul(CList,A,2)
    return A

#########################################################################################################
## CNOT Synthesis -  Gaussian Elimination
#########################################################################################################

def CNOT_GaussianElim(A):
    '''Gaussian Elimination CNOT Circuit Synthesis'''
    A = A.T.copy()
    opList = []
    m,n = A.shape
    r,c = 0,0
    while r < m and c < n:
        rList = [j for j in range(r,m) if A[j,c]==1]
        if len(rList) > 0:
            j = rList.pop(0)
            if j > r:
                A[r] ^= A[j]
                opList.append((j,r))
            for j in [j for j in range(m) if A[j,c]==1]:
                if j != r:
                    A[j] ^= A[r]
                    opList.append((r,j))
            r+=1
        c+=1
    opList.reverse()
    ## should be identity permutation
    ix = permMat2ix(A)
    return ix, opList

def CNOT_GaussianElimNoSwap(A):
    '''Gaussian Elimination CNOT Circuit Synthesis'''
    A = A.T.copy()
    opList = []
    m,n = A.shape
    r,c = 0,0
    while r < m and c < n:
        rList = [j for j in range(r,m) if A[j,c]==1]
        if len(rList) > 0:
            j = rList.pop(0)
            if j > r:
                A[r] ^= A[j]
                opList.append((j,r))
            for j in [j for j in range(m) if A[j,c]==1]:
                if j != r:
                    A[j] ^= A[r]
                    opList.append((r,j))
            r+=1
        c+=1
    opList.reverse()
    ## should be identity permutation
    ix = permMat2ix(A)
    return ix, opList

#########################################################################################################
## CNOT Synthesis - from Optimal Synthesis of Linear Reversible Circuits 
#########################################################################################################

def CNOT_Patel(A,useSWAP=True):
    '''Patel - asymptotically optimal CNOT Synthesis'''
    A = A.T.copy()
    n = len(A)
    ## round((log2 n)/2
    m = max(int(np.round(np.log2(n)/2)),2)
    A, opList1 = CNOT_Synth_lwr(A, m,useSWAP)
    # print('CNOT_Synth_lwr(A)')
    # print(ZMatPrint(A))
    At, opList2 = CNOT_Synth_lwr(A.T, m,useSWAP)
    # print('CNOT_Synth_lwr(A.T)')
    # print(ZMatPrint(A))
    opList = (opList1 + opListT(opList2))
    opList = opListInv(opList)
    opList = QPerm2Front(opList)
    # ix = permMat2ix(At.T)
    return opList

def CNOT_Synth_lwr(A, m,useSWAP=True):
    opList = []
    n = len(A)
    # print(m,n,n//m, (n//m)*m)
    for k in range((n-1)//m + 1):
        a = k*m
        b = min((k+1)*m,n)
        for i in range(a,n-1):
            if np.sum(A[i,a:b]) > 0:
                B = A ^ A[i]
                for j in range(i+1,n):
                    if np.sum(B[j,a:b]) == 0:
                        A[j] ^= A[i]
                        opList.append(('CX',(i,j)))
        for c in range(a,b):
            rList = []
            for r in range(c,n):
                if A[r,c] == 1:
                    rList.append(r)
            j = rList.pop(0)
            if j > c:
                ## Swap cols
                if useSWAP:
                    ix = np.arange(n)
                    ix[j],ix[c] = ix[c],ix[j]
                    opList.append(('QPerm',tuple(ix)))
                    A = A[ix]
                else:
                    opList.append(('CX',(j,c)))
                    opList.append(('CX',(c,j)))
                    A[c] ^= A[j]
                    A[j] ^= A[c]
            for j in rList:
                ## eliminate entries below c
                opList.append(('CX',(c,j)))
                A[j] ^= A[c]
    return A, opList


#########################################################################################################
## CNOT Synthesis - greedy
#########################################################################################################

def matWt(A):
    # sorted weights of columns and rows, returned as tuple for sorting
    A = ZMat(A)
    sA = tuple(sorted(vecJoin(matColSum(A),matColSum(A.T))))
    return sA

def CNOT_greedy(A,verbose=False):
    A = A.T.copy()
    m,n = A.shape
    done = (np.sum(A) == len(A))
    opList = []
    stepCount = 1
    if verbose:
        print(ZMatPrint(A))
        print('Sorted Row/Col Weights:',matWt(A))
    while not done:
        # ops = []
        minOp = None
        minB = None
        for i in range(m):
            for j in range(m):
                if i != j:
                    B = A.copy()
                    B[i] ^= B[j]
                    w = matWt(B)
                    op = (w,j,i)
                    if minOp is None or minOp > op:
                        minOp = op
                        minB = B
        w,j,i = minOp
        A = minB
        opList.append((j,i))
        if verbose:
            print('Step ',stepCount, ': Apply $\\textit{CNOT}_{',j,i,'}$')
            print(ZMatPrint(A))
            print('Sorted Row/Col Weights:',matWt(A))
            stepCount+=1
        done = (np.sum(A) == len(A))
    opList.reverse()
    ix = permMat2ix(A)
    if verbose:
        print(f'Qubit permutation: {ix}')
    return ix,opList

def CNOT_greedy_depth(A,verbose=False):
    A = A.T.copy()
    d = 0
    m,n = A.shape
    done = (np.sum(A) == len(A))
    opList = []
    stepCount = 1
    if verbose:
        print(ZMatPrint(A))
        print('Sorted Row/Col Weights:',matWt(A))
    while not done:
        ops = []
        S = np.arange(n)
        wLast = tuple([n]*(2*n))
        d+=1
        while len(S) > 1:
            for (i,j) in iter.combinations(S,2):
                for k in range(2):
                    B = A.copy()
                    B[i] ^= B[j]
                    w = matWt(B)
                    ops.append((w,j,i))
                    i,j = j,i
            (w,j,i) = min(ops)
            if w < wLast:
                A[i] ^= A[j]
                opList.append((j,i))
                wLast = w
                S = set(S) - {i,j}
                if verbose:
                    print('Step ',stepCount, ': Apply $\\textit{CNOT}_{',j,i,'}$')
                    print(ZMatPrint(A))
                    print('Sorted Row/Col Weights:',matWt(A))
                    stepCount+=1
            else:
                S = []
        done = (np.sum(A) == len(A))
    opList.reverse()
    ix = permMat2ix(A)
    if verbose:
        print(f'Qubit permutation: {ix}')
    print(f'd={d}')
    return ix,opList

##############################################################################
## Transvection decomposition of Symplectic Matrix
##############################################################################

def Fmat(U,i,j):
    '''Return F-matrix: U_{i,j} & U_{i,j+n}\\U_{i+n,j} & U_{i+n,j+n}'''
    m,n = symShape(U)
    F = ZMatZeros((2,2))
    for r in range(2):
        for c in range(2):
            F[r,c] = U[i + m*r, j + n*c]
    return F

def Fdet(F):
    '''determinant of 2x2 binary matrix'''
    return (F[0,0] * F[1,1]) ^ (F[0,1] * F[1,0])

def FRk(U):
    '''Rank of 2x2 Binary Matrix U'''
    if np.sum(U) == 0:
        return 0
    if Fdet(U) == 0:
        return 1
    return 0

def symR2(U):
    '''return an nxn binary matrix such that S[i,j] = 1 if F_ij is invertible or 0 otherwise'''
    m,n = symShape(U)
    ## we calculate the determinant in parallel: U_XX U_ZZ + U_XZ U_ZX
    UR2 = (U[:m,:n] & U[m:,n:])
    UR2 ^= (U[:m,n:] & U[m:,:n])
    return UR2

def symR0(U):
    '''return an nxn binary matrix such that S[i,j] = 1 if F_ij is zero'''
    m,n = symShape(U)
    ## Flip 0 and 1
    U = 1 ^ U
    ## all zero entries have 1 in all four of U_XX U_XZ U_ZX U_ZZ so multiply together these matrices
    UR0 = (U[:m,:n] & U[m:,n:]) & (U[:m,n:] & U[m:,:n])
    return UR0

def symR1(UR2,UR0):
    '''return an nxn binary matrix such that S[i,j] = 1 if F_ij is rank 1'''
    ## S_ij=1 if either F_ij is rank 2 or rank 0
    UR1 = (UR2 ^ UR0)
    ## Flipping 0 and 1 results in F_ij rank 1
    UR1 ^= 1
    return UR1

def FMatInv(F):
    '''Fast method for calculating inverse of 2x2 binary matrix just swap U_XX and U_ZZ entries'''
    temp = F.copy()
    temp[0,0],temp[1,1] = F[1,1],F[0,0]
    return temp
    
#########################################################################################################
## Volanto Transvection algorithm 
#########################################################################################################

def ElimRk1(Fj,Fk):
    '''For Fj invertible and Fk rank one, return transvection T1jk which sets Fk to zero'''
    ## Fii inverse - this may change during elimination process
    FjInv = FMatInv(Fj)
    ## calculate a,b,c,d for transvection
    FjFk = matMul(FjInv,Fk,2)
    ## row weights - reverse order of a and b
    b,a = np.sum(FjFk,axis=-1)
    ## col weights
    c,d = np.sum(FjFk,axis=0)
    # project any non-zero elements to 1
    return tuple(mod1([a,c,b,d]))

def ElimRk2(Fj,Fk):
    '''For Fj, Fk invertible, return transvection T2jk which makes both rank 1'''
    (a,b) = Fj[0]
    (c,d) = Fk[1]
    return (a,c,b,d)

def csynth_volanto(U):
    '''Decomposition of symplectic matrix U into 2-transvections, SWAP and single-qubit Clifford layers'''
    ## we will reduce UC to single-qubit Clifford layer
    UC = U.copy()
    m,n = symShape(UC)
    ## list of 2-transvections
    vList = []
    mList = set(np.arange(n))
    for i in range(m):
        ## invertible F matrices in row i
        invList = [j for j in mList if Fdet(Fmat(UC,i,j)) > 0]
        # print(func_name(),invList,UC)
        if len(invList) > 0:
            ## a is smallest j such that Fji is invertible
            a = invList.pop(0)
            mList.remove(a)
            ## ensure that Fii is the only invertible matrix in col i by pairing invertible matrices in row j,k
            for r in range(len(invList)//2):
                j = invList[2*r]
                k = invList[1+2*r]
                acbd = ElimRk2(Fmat(UC,i,j),Fmat(UC,i,k))
                UC = applyTv2(UC,acbd,(j,k))
                vList.append((acbd,(j,k)))
        ## eliminate rank 1 F matrices in column i
        for j in mList:
            Fij = Fmat(UC,i,j)
            if np.sum(Fij) > 0:
                acbd = ElimRk1(Fmat(UC,i,a),Fij)
                UC = applyTv2(UC,acbd,(a,j))
                vList.append((acbd,(a,j)))
    return opListInv(vList),UC

#########################################################################################################
## Brugiere greedy algorithm
#########################################################################################################

def LU(A,i,j,k):
    n = len(A)
    B = A.copy()
    CXList = []
    ixL = np.arange(n)
    if i != k:
        ixL[i],ixL[k] = ixL[k],ixL[i]
        B = B[ixL]
    ixR = np.arange(n)
    if j != k:
        ixR[j],ixR[k] = ixR[k],ixR[j]
        B = B[:,ixR]
    # print('B',i,j,k)
    # print(ZMatPrint(B))

    for l in range(k):
        if B[k,l] == 1:
            B[:,l] ^= B[:,k]
            CXList.append(('CX',(k,l)))
    # print('B LU')
    # print(ZMatPrint(B))
    return B,ixL,CXList,ixR

def LUDecompMinCost(A):
    preOps,postOps = [],[]
    n = len(A)
    for k in range(n-1,-1,-1):
        iList,jList = np.nonzero(A[:k+1,:k+1])
        BMin = None
        GList = None
        # print('A',k)
        # print(ZMatPrint(A))
        for i,j in zip(iList,jList):
            B,ixL,CXList,ixR = LU(A,i,j,k)
            BCurr = (len(CXList),matSum(B))
            if BMin is None or BCurr < BMin:
                BMin = BCurr
                GList = B,ixL,CXList,ixR
        A,ixL,CXList,ixR = GList
        preOps = [('QPerm',tuple(ixL))] + preOps
        postOps = postOps + [('QPerm',tuple(ixR))] + CXList
    return A, preOps, postOps


def LUDecompSparse(A):
    preOps,postOps = [],[]
    n = len(A)
    for k in range(n-1,-1,-1):
        iList,jList = np.nonzero(A[:k+1,:k+1])
        BMin = None
        ij = None
        # print('A',k)
        # print(ZMatPrint(A))
        for i,j in zip(iList,jList):
            BCurr = matSum(A[i]) + matSum(A[:,j])
            if BMin is None or BCurr < BMin:
                BMin = BCurr
                ij = (i,j)
        i,j = ij
        A,ixL,CXList,ixR = LU(A,i,j,k)
        preOps = [('QPerm',tuple(ixL))] + preOps
        postOps = postOps + [('QPerm',tuple(ixR))] + CXList
    return A, preOps, postOps

def LUDecompStd(A):
    preOps,postOps = [],[]
    n = len(A)
    for k in range(n-1,-1,-1):
        iList,jList = np.nonzero(A[:k+1,:k+1])
        i,j = iList[-1],jList[-1]
        A,ixL,CXList,ixR = LU(A,i,j,k)
        preOps = [('QPerm',tuple(ixL))] + preOps
        postOps = postOps + [('QPerm',tuple(ixR))] + CXList
    return A, preOps, postOps

def CNOTBrug(A,LUmethod='MinCost'):
    if LUmethod=='MinCost':
        B, OpL,OpR = LUDecompMinCost(A)
    elif LUmethod=='Sparse':
        B, OpL,OpR = LUDecompSparse(A)
    elif LUmethod=='Std':
        B, OpL,OpR = LUDecompStd(A)
    C, OpC = BrugLower(B)
    # n = len(A)
    # OL = opList2sym(OpL,n)[:n,:n]
    # OR = opList2sym(OpR,n)[:n,:n]
    # OC = opList2sym(OpC,n)[:n,:n]
    # A1 = matMul( matMul(OL,A,2), matMul(OR,OC,2),2)
    # print('In')
    # print(ZMatPrint(A1))
    opList =  opListInv(OpR + OpC + OpL)
    return QPerm2Front(opList)

def USum(A):
    A = AUT(A,True)
    return matSum(A)

def AUT(A,exDiag=False):
    temp = ZMatZeros(A.shape)
    b = 1 if exDiag else 0
    for i in range(len(A)-b):
        temp[i,i+b:] = A[i,i+b:]
    return temp

def BrugLower(A):
    m,n = A.shape
    opList = []
    A = A.copy()
    while not USum(A) == 0:
        i,j = bin2Set(SelectRowOperation(A))
        opList.append(('CX',(i,j)))
        A[:,j] ^= A[:,i]
    # opList.reverse()
    return A, opList

def SelectRowOperation(A):
    m,n = A.shape
    # A = AUT(A)
    j = 0
    S = ZMat([1]*n)
    while matSum(S) > 2:      
        a = A[j]  
        S0 = (1 ^ a) & S
        S1 = a & S
        S = S0 if matSum(S1) < 2 else S1
        j += 1
    return S

#########################################################################################################
## MW greedy algorithm
#########################################################################################################

def csynth_greedy(U,params):
    '''Decomposition of symplectic matrix U into 2-transvections, SWAP and single-qubit Clifford layers'''
    ## we will reduce UC to single-qubit Clifford layer
    mode = params.mode
    m,n = symShape(U)
    U = U.copy()
    # n = len(U)//2
    ## list of 2-transvections
    opList = []
    h,w = TvWt(U,params) if mode=='sym' else CNOTWt(U,params)
    done = (h < 0.00001)
    currWait = 0
    lastMin = None
    dLast = 0
    oLast = None
    hix = 1 if params.hv else 0
    while not done:
        oMin = None
        oU = None
        opts = TvOptions(U) if mode == 'sym' else CNOTOptions(U)
        # print('oLast',oLast)
        for opType,ij in opts:
            UTv = applyOp(U,opType,ij) 
            h,w = TvWt(UTv,params) if mode=='sym' else CNOTWt(UTv,params)
            o = (w,h,(opType,ij)) if params.hv else (h,w,(opType,ij))
            if oLast is not None and o > oLast:
                d = dLast + 10000
            elif params.minDepth:
                d = len(opListLayers(opList + [(opType,ij)])) 
            else:
                d = dLast
            oCurr = (d,o)
            if (oMin is None) or (oMin > oCurr):
                oMin = oCurr 
                oU = UTv
        if lastMin is not None and oMin >= lastMin:
            currWait+=1
        else:
            currWait = 0
            lastMin = oMin 
        # print('currWait',currWait,'lastMin',lastMin)
        if (params.wMax > 0 and currWait > params.wMax):
            # print('params.wMax exceeded')
            return [],np.arange(n),[]
        # w,h,Tv = oMin
        # print(oMin)
        dLast, oLast = oMin
        opList.append(oLast[-1])
        U = oU
        done = (oLast[hix] < 0.00001)
    # ix,CList = UC2ixCList(U)
    opList = opListInv(opList)
    return opList,U

def TvOptions(U):
    '''default ijOptions'''
    m,n = symShape(U)
    ijList = set()
    UR2 = symR2(U)
    UR0 = symR0(U)
    UR1 = symR1(UR2,UR0)
    for i in range(m):
        R2 = bin2Set(UR2[i])
        R1 = bin2Set(UR1[i])
        L = len(R2)
        for j in range(L-1):
            for k in range(j+1,L):
                ijList.add((R2[j],R2[k]))
        for j in R2:
            for k in R1:
                ijList.add((j,k))
    vList = {(a % 2,b%2,a//2,b//2) for a in range(1,4) for b in range(1,4)}
    return {(v,ij) for v in vList for ij in ijList}

def CNOTOptions(A,allOpts=False):
    '''default ijOptions'''
    m,n = symShape(A)
    # return [(i,j) for i in range(n) for j in range(n) if i != j]
    U = A[:m,:n]
    if allOpts:
        return [('CX',(i,j)) for i in range(n) for j in range(n) if i!=j]
    ## dot product of columns with columns - non-zero elements have overlap and so are in the list
    iList,jList = np.nonzero((U.T @ U)) 
    ## exclude those along the diagonal
    return [('CX',(i,j)) for (i,j) in zip(iList,jList) if i != j]

def overlapWt(A):
    return ZMat([(matSum(x ^ A)-np.sum(x)) for x in A])

def TvWt(U,params):
    hi,ht,hl,hr = params.hi,params.ht,params.hl,params.hr
    m,n = symShape(U)
    ## Invertible 2x2 matrices
    UR2 = symR2(U)
    ## All zero 2x2 matrices
    UR0 = symR0(U)
    ## Rank 1 2x2 matrices - not U1 and not U2
    UR1 = symR1(UR2,UR0)
    # URn = UR2 * n + UR1
    c1 = vecJoin(matColSum(UR1),matColSum(UR1.T)) if ht else matColSum(UR1)
    c2 = vecJoin(matColSum(UR2),matColSum(UR2.T)) if ht else matColSum(UR2)
    if hl:
        h = hr * np.sum(np.log(c1 + c2))/len(c1)
    else:
        h = hr * (matSum(UR1) + matSum(UR2) - n)/n
    return h, tuple(sorted(c2 * n + c1))

def CNOTWt(U,params):
    m,n = symShape(U)
    if params.hi == 0:
        U = U[:m,:n]
    sA = vecJoin(matColSum(U),matColSum(U.T)) if params.ht else matColSum(U)
    Ls = len(sA) if params.hl else len(U)
    h = params.hr * matSum(np.log(sA))/Ls if params.hl else params.hr * (matSum(U) - Ls)/Ls
    return h, tuple(sorted(sA))

#########################################################################################################
## MW Astar algorithm 
#########################################################################################################

def synth_astar(U,params):
    '''astar circit synthesis'''
    ## can choose from various decomposition methods
    return synth_treap(U,params)

def synth_treap(U,params):
    '''Astar using treap to manage size of priority queue
     no updating of elements of queue - lower memory requriement'''
    # m,n = symShape(U)
    mode,qMax = params.mode,params.qMax
    Q = treap.treap()
    Utup = hash(tuple(U.ravel()))
    currId = 0
    visited = {Utup: currId}
    g = 0
    h, w = CNOTWt(U,params) if mode=='GL2' else TvWt(U,params)
    Aid = -1
    op = ((0,0,0,0),(0,0),0,0)
    DB = [(Aid,op,g,h)]
    Q[(g+h,g,h,w,op)] = currId
    while Q.length > 0:
        s,Aid = Q.remove_min()
        _,_,g,h = DB[Aid]
        opList = getOpListRec(Aid,DB)
        A = opListApply(opList,U)
        if (h < 0.000001):
            return opListInv(opList),A
        # g = g + 1
        myOpts = CNOTOptions(A) if mode=='GL2' else TvOptions(A)
        for opType,ij in myOpts:
            Ui = applyOp(A,opType,ij)
            Utup = hash(tuple(Ui.ravel()))
            if Utup not in visited:
                h, w = CNOTWt(Ui,params) if mode=='GL2' else TvWt(Ui,params)
                currId += 1
                visited[Utup] = currId
                op = (opType,ij,0,0)
                gi = len(opListLayers(opList + [(opType,ij)])) if params.minDepth else g + 1
                DB.append((Aid,op,gi,h))
                Q[(gi+h,gi,h,w,op)] = currId
        if Q.length > qMax:
            for i in range(qMax,Q.length):
                Q.remove_max()
    return None

#########################################################################################################
## Qiskit Synthesis
#########################################################################################################

def qiskit2opList(circ):
    '''convert qiskit circuit to opList'''
    opList = []
    for op in circ.data:
        opName = op.operation.name.upper()
        qList = [q._index for q in op.qubits]
        opList.append((opName, qList)) 
    return opList


def csynth_qiskit(qc,method='greedy'):
    '''qiskit synthesis - various methods
    input is qiskit circuit qc''' 
    if method == 'ag':
        return qiskit.synthesis.synth_clifford_ag(qc)
    elif method == 'layers':
        return qiskit.synthesis.synth_clifford_layers(qc)
    else:
        return qiskit.synthesis.synth_clifford_greedy(qc)

#########################################################################################################
## from TU Munich QMAP - A tool for Quantum Circuit Compilation
## https://github.com/cda-tum/mqt-qmap Depth-Optimal Synthesis of Clifford Circuits with SAT Solvers
#########################################################################################################

def csynth_SAT(S):
    '''SAT synthesis - warning slow for n>5
    input is qiskit clifford S'''
    # S = qmap.Tableau(S)
    # qc_alt, syn_res = qmap.synthesize_clifford(target_tableau=S)
    qc_alt, syn_res = qmap.optimize_clifford(S.to_circuit())
    return qiskit2opList(qc_alt)


#########################################################################################################
## PyZX
#########################################################################################################

def csynth_pyzx(mytext):
    zxcircuit = pyzx.Circuit.from_qasm(mytext)
    zxg = zxcircuit.to_graph()
    pyzx.simplify.full_reduce(zxg)
    c1=pyzx.extract_circuit(zxg)
    qc = qiskit.QuantumCircuit.from_qasm_str(c1.to_qasm())
    return qiskit2opList(qc)

#########################################################################################################
## Quantinuum tket: https://docs.quantinuum.com/tket/
## Various optimisation algorithms - best seems to be FullPeepholeOptimise
#########################################################################################################

def tket2opList(circ):
    '''convert tket circuit to opList'''
    opList = []
    for op in circ.get_commands():
        opList.append((str(op.op),[q.index[0] for q in op.qubits]))
    return opList


def csynth_tket(mytext,option='FullPeepholeOptimise',nReps=1):
    '''tket synthesis - various options'''
    
    qc = pytket.qasm.circuit_from_qasm_str(mytext)
    for i in range(nReps):
        if option=='CliffordSimp':
            pytket.passes.CliffordSimp().apply(qc)
        elif option=='SynthesiseTket':
            pytket.passes.SynthesiseTket().apply(qc)
        elif option=='CliffordResynthesis':
            pytket.passes.CliffordResynthesis().apply(qc)
        elif option=='FullPeepholeOptimise':
            pytket.passes.FullPeepholeOptimise().apply(qc)
        else:
            pytket.passes.DecomposeBoxes().apply(qc)
            pytket.passes.PauliSimp().apply(qc)
            pytket.passes.FullPeepholeOptimise().apply(qc)

    # return opList
    opList = tket2opList(qc)
    return opList


def csynth_tket(mytext,option='FullPeepholeOptimise',nReps=10):
    '''tket synthesis - various options'''
    
    qc = pytket.qasm.circuit_from_qasm_str(mytext)
    for i in range(nReps):
        if option=='CliffordSimp':
            pytket.passes.CliffordSimp().apply(qc)
        elif option=='SynthesiseTket':
            pytket.passes.SynthesiseTket().apply(qc)
        elif option=='CliffordResynthesis':
            pytket.passes.CliffordResynthesis().apply(qc)
        else:
            pytket.passes.FullPeepholeOptimise().apply(qc)
    # return opList
    opList = tket2opList(qc)
    return opList

#########################################################################################################
## Stim
#########################################################################################################

def stim2opList(qc):
    '''convert stim circuit to opList form'''
    opList = []
    for op in qc:
        opData = str(op).split(" ")
        opName = opData[0]
        qList = [int(q) for q in opData[1:]]
        ## Single Qubit Gates
        if opName in {'I','X','Y','Z','C_XYZ','C_ZYX','H','H_XY','H_XZ','H_YZ','SQRT_X','SQRT_X_DAG',
                      'SQRT_Y','SQRT_Y_DAG','SQRT_Z','SQRT_Z_DAG','S','S_DAG',
                      'M','MR','MRX','MRY','MRZ','MX','MY','MZ','R','RX','RY','RZ'}:
            for q in qList:
                opList.append((opName,[q]))
        ## 2-Qubit Gates
        elif opName in {'CNOT','CX','CXSWAP','CY','CZ','CZSWAP','ISWAP','ISWAP_DAG','SQRT_XX','SQRT_XX_DAG','SQRT_YY',
                        'SQRT_YY_DAG','SQRT_ZZ','SQRT_ZZ_DAG','SWAP','SWAPCX','SWAPCZ','XCX','XCY','XCZ',
                        'YCX','YCY','YCZ','ZCX','ZCY','ZCZ'
                        'MXX','MYY','MZZ'}:
            for i in range(len(qList)//2):
                opList.append((opName,[qList[2*i],qList[2*i+1]]))
        ## Multi Qubit Gates
        else:
            opList.append((opName,qList))
    return opList

def csynth_stim(U):
    '''Stim synthesis - not optimised for 2-qubit gate count'''
    submethods = ['elimination','graph_state']
    xx,xz,zx,zz = sym2components(U)
    T = stim.Tableau.from_numpy(x2x=xx,x2z=xz,z2x=zx,z2z=zz)
    qc = T.to_circuit(method=submethods[0])
    return stim2opList(qc)

##########################################################
## rustiq
##########################################################

def sym2PauliStr(U):
    '''convert symplectic matrix to Pauli strings for rustiq'''
    m,n = symShape(U)
    pauli_list = ['I','X','Z','Y']
    ## Zi first, then Xi
    # Uint = ZMatVstack([U[m:],U[:m]])
    # Uint = Uint[:,:n] + 2*Uint[:,n:] 
    Uint = U[:,:n] + 2*U[:,n:] 
    return ["".join([pauli_list[a] for a in myRow]) for myRow in Uint]

def csynth_rustiq(U,iter=10):
    stabilisers = sym2PauliStr(U)
    return rustiq.clifford_synthesis(stabilisers,rustiq.Metric.COUNT, syndrome_iter=iter)