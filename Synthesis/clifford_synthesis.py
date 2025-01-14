import add_parent_dir
from common import *
from NHow import *
from CliffordOps import *
import numpy as np
import qiskit, qiskit.circuit, qiskit.qasm2
from mqt import qmap
import stim
import csv
import treap
import sys
import os
## commented out as these don't compile on the cluster
import pytket, pytket.tableau, pytket.passes, pytket.qasm


######################################################
## Random Clifford Generation
######################################################

def bin2trans(stabs,destabs):
    '''convert stabs and detabs to a permutation, series of single-qubit Cliffords and 2-qubit transvections'''
    n = len(stabs)
    ix = np.arange(n)
    CList = []
    vList = []
    for i in range(n):
        U = ZMatZeros((2,2*n))
        U[0,i:n] = stabs[i][:n-i]
        U[0,n+i:] = stabs[i][n-i:]
        U[1,i:n] = destabs[i][:n-i]
        U[1,n+i:] = destabs[i][n-i:]        
        vListi,ixi,CListi = csynth_voltano(U)
        vList.extend(reversed(vListi))
        j = ixi[0]
        # print(func_name(),i,j)
        if j > i:
            ix[[i,j]] = ix[[j,i]]
        CList.append(CListi[0])
    return vList, ix, CList
        
def bin2sym(stabs,destabs):
    '''convert stabs, destabs to symplectic matrix'''
    vList, ix, CList = bin2trans(stabs,destabs)
    return trans2sym(vList, ix, CList)

def randomCliffordBin(rng, r):
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

def symRandom(rng,n):
    '''generate random Clifford operator on n qubits'''
    stabs, destabs = randomCliffordBin(rng, n)
    vList, ix, CList = bin2trans(stabs,destabs)
    U = ZMat(trans2sym(vList, ix, CList))
    return U

def GL2random(rng,n):
    '''generate random nxn invertible matrix'''
    xList = randomGLbin(rng,n)
    return bin2GL(xList)

def CXTest(U,ix,CXList):
    A = CNOT2Mat(ix,CXList)
    return np.sum(U ^ A) == 0

def CNOT2Mat(ix,CXList):
    ## start with permutation matrix
    A = permMat(ix)
    ## CNOTs represent column operations 
    ## add col i to row j
    for (i,j) in CXList:
        A[:,j] ^= A[:,i]
    return A

#########################################################################################################
## Utilities
#########################################################################################################

def CNOT2opList(ix,opList):
    return [('QPerm',ix)] + [('CX',[i,j]) for (i,j) in opList]

def transTest(U,vList,ix,CList):
    U2 = trans2sym(vList,ix,CList)
    return np.sum(U ^ U2) == 0

def sym2qc(U):
    '''Convert symplectic matrix U to a qiskit circuit object'''
    return qiskit.quantum_info.Clifford(U).to_circuit()

def qc2qasm(qc):
    '''convert qiskit circuit object to qasm 2 string'''
    return qiskit.qasm2.dumps(qc)

def readMatFile(fileName):
    f = open(fileName)
    mytext = f.read()
    mytext = mytext.split('\n')
    temp = []
    for s in mytext:
        if len(s) > 0:
            A = bin2ZMat(s)[0]
            n = int(np.round(np.sqrt(len(A))))
            temp.append(np.reshape(A,(n,n)))
    return temp

def bravyi_run(circuits,i,params):
    '''Run optimisation for circuits in Bravyi et al'''
    circuitName, mytext = circuits[i]

    ## make a Qiskit quantum circuit and Clifford from saved text
    qc = qiskit.QuantumCircuit.from_qasm_str(mytext)
    S = qiskit.quantum_info.Clifford(qc)
    U = ZMat(S.symplectic_matrix)
    U1 = U.copy()

    ## starting time
    sT = currTime()

    ## PyTket
    if params.method == 'pytket':
        opList = csynth_tket(mytext,params.methodName)

    ## Qiskit
    elif params.method == 'qiskit':
        circ = csynth_qiskit(S,params.methodName)
        opList = qiskit2opList(circ)

    ## Voltano
    elif params.method == 'voltano':
        vList, ix, CList = csynth_voltano(U)
        opList = trans2opList(vList,ix,CList)

    ## Greedy Algorithm
    elif params.method == 'greedy':
        vList, ix, CList = csynth_greedy(U)
        opList = trans2opList(vList,ix,CList)

    ## Astar
    elif params.method == 'astar':
        vList, ix, CList = csynth_astar(U,params.r1,params.r2,params.qMax)
        opList = trans2opList(vList,ix,CList)
        
    ## STIM
    elif params.method == 'stim':
        opList = csynth_stim(U)
    
    ## if no method specified, just count gates in input circuit
    else:
        opList = qiskit2opList(qc)

    ## write results to file
    f = open(params.outfile,'a')
    r = entanglingGateCount(opList)
    t = currTime()-sT
    c = opList2str(opList,ch=" ")
    if params.method in ['voltano','greedy','astar']:
        check = transTest(U1,vList,ix,CList)
    else:
        check = ""
    if params.astarRange:
        f.write(f'{i+1}\t{circuitName}\t{params.r1}\t{params.r2}\t{r}\t{t}\t{check}\t{c}\n')
    else:
        f.write(f'{i+1}\t{circuitName}\t{r}\t{t}\t{check}\t{c}\n')
    f.close()
    ## return result + exec time + opList
    return (i,circuitName,r,t,c)


def random_run(UList,i,params):
    '''Run optimisation for circuits in Bravyi et al'''
    U = UList[i]
    
    ## convert GL2 to Symplectic
    if params.t == 'GL2':
        U = symCNOT(U)

    m,n = symShape(U)
        
    ## make a Qiskit quantum circuit and Clifford from saved text
    # qc = qiskit.QuantumCircuit.from_qasm_str(mytext)
    # S = qiskit.quantum_info.Clifford(qc)
    # U = ZMat(S.symplectic_matrix)
    qc = sym2qc(U)
    mytext = qc2qasm(qc)

    U1 = U.copy()
    ## starting time
    sT = currTime()


    ## PyTket
    if params.method == 'pytket':
        opList = csynth_tket(mytext,params.methodName)

    elif params.method == 'CNOT_gaussian':
        ix,CXList = CNOT_GaussianElim(U[:n,:n])
        opList = CNOT2opList(ix,CXList)

    elif params.method == 'CNOT_Patel':
        ix,CXList = CNOT_Patel(U[:n,:n])
        opList = CNOT2opList(ix,CXList)

    elif params.method == 'CNOT_greedy':
        ix,CXList = CNOT_greedy(U[:n,:n])
        opList = CNOT2opList(ix,CXList)

    elif params.method == 'CNOT_depth':
        ix,CXList = CNOT_greedy_depth(U[:n,:n])
        opList = CNOT2opList(ix,CXList)

    ## SAT
    elif params.method == 'sat':
        C = qiskit.quantum_info.Clifford(qc)
        opList = csynth_SAT(C)

    ## Qiskit
    elif params.method == 'qiskit':
        C = qiskit.quantum_info.Clifford(qc)
        circ = csynth_qiskit(C,params.methodName)
        opList = qiskit2opList(circ)

    ## Voltano
    elif params.method == 'voltano':
        vList, ix, CList = csynth_voltano(U)
        opList = trans2opList(vList,ix,CList)

    ## Greedy Algorithm
    elif params.method == 'greedy':
        vList, ix, CList = csynth_greedy(U)
        opList = trans2opList(vList,ix,CList)

    ## Astar
    elif params.method == 'astar':
        vList, ix, CList = csynth_astar(U,params.r1,params.r2,params.qMax)
        opList = trans2opList(vList,ix,CList)
        
    ## STIM
    elif params.method == 'stim':
        opList = csynth_stim(U)
    
    ## if no method specified, just count gates in input circuit
    else:
        opList = qiskit2opList(qc)

    ## write results to file
    f = open(params.outfile,'a')
    r = entanglingGateCount(opList)
    t = currTime()-sT
    c = opList2str(opList,ch=" ")
    if params.method in ['voltano','greedy','astar']:
        check = transTest(U1,vList,ix,CList)
    elif params.method in ['CNOT_greedy','CNOT_depth','CNOT_gaussian','CNOT_Patel']:
        check = CXTest(U1[:n,:n],ix,CXList)
    else:
        check = ""
    if params.astarRange:
        f.write(f'{i+1}\t{n}\t{params.r1}\t{params.r2}\t{r}\t{t}\t{check}\t{c}\n')
    else:
        f.write(f'{i+1}\t{n}\t{r}\t{t}\t{check}\t{c}\n')
    f.close()
    ## return result + exec time + opList
    return (i,r,t,c)

def set_global_params(params):
    '''Process parameters - set name of output file'''
    mydate = time.strftime("%Y%m%d-%H%M%S")
    ## for pytket, qiskit set methodName
    if params.method == 'pytket':
        methods = ['FullPeepholeOptimise','CliffordSimp','SynthesiseTket','CliffordResynthesis']
        params.methodName = methods[params.submethod]
    elif params.method == 'qiskit':
        methods = ['greedy','ag']
        params.methodName = methods[params.submethod]
    ## for astar, record r1, r2, qmax
    if params.method == 'astar':
        myfile = f"{params.file}-{params.method}-r{params.r1}-{params.r2}-q{params.qMax}-{mydate}.txt"
    elif params.method in {'qiskit','pytket'}:
        myfile = f"{params.file}-{params.method}-{params.methodName}-{mydate}.txt"
    else:
        myfile = f"{params.file}-{params.method}-{mydate}.txt"
    if params.astarRange:
        myfile = f"{params.file}-{params.method}-range-q{params.qMax}-{mydate}.txt"
    cwd = os.getcwd()
    params.outfile = f'{cwd}/{myfile}'
    

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

def entanglingGateCount(opList):
    '''count number of entangling gates in list of operataors opList'''
    c = 0
    for opName,qList in opList:
        ## ignore SWAP and QPerm gates
        if len(qList) > 1 and opName != 'SWAP' and opName != "QPerm":
            c += 1
    return c

def opList2str(opList,ch="\n"):
    '''convert oplist to string rep'''
    temp = []
    for opName,qList in opList:
        qStr = ",".join([str(q) for q in qList])
        temp.append(f'{opName}:{qStr}')
    return ch.join(temp)

           
def mat2ixCList(UC):
    '''Convert binary matrix with exactly one Fij in each row/col of rank 2
    to a permutation and list of single-qubit Cliffords'''
    ## extract qubit permutation
    UR2 = symR2(UC)
    ix = permMat2ix(UR2)
    # print(func_name(),ix,UC,UR2)
    ## extract list of single-qubit cliffords
    CList =  [Fmat(UC,i,ix[i]) for i in range(len(ix))]
    return ix, CList

def trans2opList(vList,ix,UC):
    '''Convert output of MW optimisations to opList
    vList: list of 2-qubit transvections
    ix: qubit permutataion
    UC: list of single-qubit Cliffords'''
    n = len(ix)
    temp = []

    ## dict for single-qubit Cliffords
    cliff_list = {'1001':'I', '0110':'H','1101':'S','1011':'SQRT_X','1110':'HS','0111':'SH'}
    for i in range(n):
        c = cliff_list[ZMat2str(UC[i].ravel())]
        ## don't add single-qubit identity operators
        if c != 'I':
            temp.append((c,[i]))
    ## check if we have the trivial permutation - if not append a QPerm operator
    if not np.all(ix == np.arange(n)):
        temp.append(('QPerm', ix))
    ## process 2-qubit transvections - these are of form \sqrt{P_1 P_2} for paulis P_i
    pauli_list = ['I','X','Z','Y']
    for acbd, ij in vList:
        acbd = ZMat(acbd)
        xz = acbd[:2] + 2 * acbd[2:]
        P = pauli_list[xz[0]] + pauli_list[xz[1]]
        temp.append((f'SQRT_{P}',ij))
    return temp

def trans2sym(vList,ixC,UC):
    '''convert list of 2-transvections, a qubit permutation and single-qubit Clifford matrix to symplectic matrix'''
    US = SymSWAP(ixC)
    UC = symKron(UC)
    U = matMul(UC,US,2)
    for acbd,ij in vList:
        U = TvMul(U,acbd,ij)
    return U

def sym2components(U):
    '''Split Symplectic matrix U into components XX,XZ,ZX,ZZ for use in stim, qiskit, pytket'''
    n = len(U)//2
    xx=np.array(U[:n,:n],dtype=bool)
    xz=np.array(U[:n,n:],dtype=bool)
    zx=np.array(U[n:,:n],dtype=bool)
    zz=np.array(U[n:,n:],dtype=bool)
    return xx,xz,zx,zz



#########################################################################################################
## CNOT Synthesis -  Gaussian Elimination
#########################################################################################################

def CNOT_GaussianElim(A):
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
    ix = permMat2ix(A)
    return ix, opList

#########################################################################################################
## CNOT Synthesis - from Optimal Synthesis of Linear Reversible Circuits 
#########################################################################################################

def CNOT_Patel(A):
    A = A.T.copy()
    n = len(A)
    ## round((log2 n)/2
    m = max(int(np.round(np.log2(n)/2)),2)
    A, opList1 = CNOT_Synth_lwr(A, m)
    At, opList2 = CNOT_Synth_lwr(A.T, m)
    opList2 = [(j,i) for (i,j) in reversed(opList2)]
    opList = (opList1 + opList2)
    opList.reverse()
    ix = permMat2ix(At.T)
    return ix, opList

def CNOT_Synth_lwr(A, m):
    opList = []
    n = len(A)
    for k in range(n//m):
        a = k*m
        b = min((k+1)*m,n)
        for i in range(a,n-1):
            if np.sum(A[i,a:b]) > 0:
                B = A ^ A[i]
                for j in range(i+1,n):
                    if np.sum(B[j,a:b]) == 0:
                        A[j] ^= A[i]
                        opList.append((i,j))
        for c in range(a,b):
            rList = []
            for r in range(c,n):
                if A[r,c] == 1:
                    rList.append(r)
            j = rList.pop(0)
            if j > c:
                ## Swap rows
                opList.append((j,c))
                opList.append((c,j))
                A[c] ^= A[j]
                A[j] ^= A[c]
            for j in rList:
                ## eliminate entries below c
                opList.append((c,j))
                A[j] ^= A[c]
    return A, opList


#########################################################################################################
## CNOT Synthesis - greedy
#########################################################################################################

def matWt(A):
    # sorted weights of columns and rows, returned as tuple for sorting
    # sA = tuple(sorted(np.hstack([np.sum(A,axis=-1),np.sum(A,axis=0)]),reverse=False))
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
                    # ops.append((w,j,i))
        # (w,j,i) = min(ops)

        # A[i] ^= A[j]
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

def TvMul(U,acbd,ij):
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

def transWt(U,r1=1,r2=1):
    ## Invertible 2x2 matrices
    UR2 = symR2(U)
    ## All zero 2x2 matrices
    UR0 = symR0(U)
    ## Rank 1 2x2 matrices - not U1 and not U2
    UR1 = symR1(UR2,UR0)
    m,n = symShape(U)

    ## Heuristic estimating number of steps required for completion
    h = (matSum(UR2) - n)/r2 + matSum(UR1)/r1
    ## Number of invertible mats in each col/row
    s2 = vecJoin(matColSum(UR2), matColSum(UR2.T))
    ## Number of rank 1 2x2 mats in each col/row
    s1 = vecJoin(matColSum(UR1), matColSum(UR1.T))
    return h, tuple(sorted((s2-n)*n + s1))

def overlapWt(A):
    return ZMat([(matSum(x ^ A)-np.sum(x)) for x in A])

def transWt(U,r1=1,r2=1):
    ## Invertible 2x2 matrices
    UR2 = symR2(U)
    ## All zero 2x2 matrices
    UR0 = symR0(U)
    ## Rank 1 2x2 matrices - not U1 and not U2
    UR1 = symR1(UR2,UR0)
    m,n = symShape(U)

    ## Heuristic estimating number of steps required for completion
    h = (matSum(UR2) - n)/r2 + matSum(UR1)/r1
    ## Number of invertible mats in each col/row
    s2 = vecJoin(matColSum(UR2), matColSum(UR2.T))
    ## Number of rank 1 2x2 mats in each col/row
    s1 = vecJoin(matColSum(UR1), matColSum(UR1.T))

    # return h, tuple(sorted((s2-n)*n + s1))
    ## second order colsum
    s3 = vecJoin(overlapWt(UR2), overlapWt(UR2.T))
    s4 = vecJoin(overlapWt(UR1), overlapWt(UR1.T))

    return h, (tuple(sorted((s2-n)*n + s1)),tuple(sorted(-s3*n^2 -s4)))
    
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

def csynth_voltano(U):
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
                UC = TvMul(UC,acbd,(j,k))
                vList.append((acbd,(j,k)))
        ## eliminate rank 1 F matrices in column i
        for j in mList:
            Fij = Fmat(UC,i,j)
            if np.sum(Fij) > 0:
                acbd = ElimRk1(Fmat(UC,i,a),Fij)
                UC = TvMul(UC,acbd,(a,j))
                vList.append((acbd,(a,j)))
    ## reverse order of transvections
    vList.reverse()
    ix, CList = mat2ixCList(UC)
    return vList,ix,CList


#########################################################################################################
## MW greedy algorithm
#########################################################################################################

def csynth_greedy(U):
    '''Decomposition of symplectic matrix U into 2-transvections, SWAP and single-qubit Clifford layers'''
    ## we will reduce UC to single-qubit Clifford layer
    U = U.copy()
    n = len(U)//2
    ## list of 2-transvections
    vList = []
    done = (np.sum(symR0(U)) == n*(n-1))
    while not done:
        oMin = None
        oU = None
        for acbd,ij in ijOptions(U):
            UTv = TvMul(U,acbd,ij)
            h,w = transWt(UTv)
            oCurr = (w,h,(acbd,ij))
            if (oMin is None) or (oMin > oCurr):
                oMin = oCurr 
                oU = UTv
        w,h,Tv = oMin
        vList.append(Tv)
        U = oU
        done = (h == 0)
    ix,CList = mat2ixCList(U)
    vList.reverse()
    return vList,ix,CList

def ijOptions(U):
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


# def ijOptions(U):
#     '''default ijOptions'''
#     m,n = symShape(U)
#     ijList = set()
#     UR12 = symR0(U) ^ 1
#     for i in range(m):
#         jkOpts = bin2Set(UR12[i])
#         L = len(jkOpts)
#         for j in range(L-1):
#             for k in range(j+1,L):
#                 ijList.add((jkOpts[j],jkOpts[k]))
#     vList = {(a % 2,b%2,a//2,b//2) for a in range(1,4) for b in range(1,4)}
#     return {(v,ij) for v in vList for ij in ijList}

#########################################################################################################
## MW Astar algorithm 
#########################################################################################################

class astarObj:
    '''object used in astar algorithm'''

    def __init__(self, g,h,parent,v,w,U):
        self.parent = parent ## parent of the object
        self.h = h ## heuristic for est number of steps remaining
        self.v = v ## transvection to get from parent to this step
        self.tWt = w ## greedy weight from transWt
        self.U = U ## hash id of the matrix U
        self.t = currTime() ## current system time
        self.updateG(g) ## update number of steps required to get to this point, plus object weight

    def __str__(self):
        '''write the object as a string'''
        return f'g={self.g}\nh={self.h}\nv={self.v}'
    
    def updateG(self,g):
        '''update the number of steps required to get to this point, plus object weight'''
        self.g = g
        self.w = (g+self.h,g,self.h,self.tWt,self.v)

    
    def getVlist(self):
        '''get list of transvections to reach this point'''
        if self.parent is not None:
            return [self.v] + self.parent.getVlist()
        return []
    
    def getUC(self,U):
        '''Reconstruct the matrix from original matrix U and transvections'''
        vList = self.getVlist()
        vList.reverse()
        UC = U.copy()
        for acbd,ij in vList:
            UC = TvMul(UC,acbd,ij)
        return UC
    
    def __eq__(self, other):
        '''equality test - compare w'''
        return self.w == other.w
    
    def __lt__(self,other):
        '''less than test - compare w'''
        return self.w < other.w

def csynth_astar(U,r1,r2,qMax):
    '''astar circit synthesis'''
    ## can choose from various decomposition methods
    res = astarTreap(U,r1,r2,qMax)
    # res = astarTreapUpdate(U,r1,r2,qMax)
    # res = astarPriQ(U,r1,r2)
    if res is not None:
        o,UC = res
        ix,CList = mat2ixCList(UC)
        ## return list of transvections, qubit permutation, plus list of single-qubit Cliffords
        return o.getVlist(),ix,CList
    return None

def astarTreap(U,r1,r2,qMax):
    '''Astar using treap to manage size of priority queue
     no updating of elements of queue - lower memory requriement'''
    Q = treap.treap()
    Utup = hash(tuple(U.ravel()))
    g = 0
    h, w = transWt(U,r1,r2)
    v = ((0,0,0,0),(0,0))
    parent = None
    o = astarObj(g,h,parent,v,w,Utup)
    visited = {Utup}
    Q[o.w] = o
    while Q.length > 0:
        w,o = Q.remove_min()
        UC = o.getUC(U)
        if (o.h == 0):
            return o,UC
        g = o.g + 2
        for acbd,ij in ijOptions(UC):
            Ui = TvMul(UC,acbd,ij)
            Utup = hash(tuple(Ui.ravel()))
            if Utup not in visited:
                hi,wi = transWt(Ui,r1,r2)
                oi = astarObj(g,hi,o,(acbd,ij),wi,0)
                visited.add(Utup)
                Q[oi.w] = oi
        if Q.length > qMax:
            for i in range(qMax,Q.length):
                Q.remove_max()
    return None

def astarPriQ(U,r1,r2):
    '''priority Queue method - no means of trimming size of queue'''
    import queue
    Q = queue.PriorityQueue()
    g = 0
    h, w = transWt(U,r1,r2)
    v = ((0,0,0,0),(0,0))
    o = astarObj(g,h,None,v,w,0)
    Q.put(o)
    Utup = hash(tuple(U.ravel()))
    visited = {Utup}
    while not Q.empty():
        o = Q.get()
        UC = o.getUC(U)
        if (o.h == 0):
            return o,UC
        g = o.g + 2
        for acbd,ij in ijOptions(UC):
            Ui = TvMul(UC,acbd,ij)
            Utup = hash(tuple(Ui.ravel()))
            if Utup not in visited:
                hi,wi = transWt(Ui,r1,r2)
                oi = astarObj(g,hi,o,(acbd,ij),wi,0)
                visited.add(Utup)
                Q.put(oi)
    return None

def astarTreapUpdate(U,r1,r2,qMax):
    '''Update parent and g when U already visited'''
    Q = treap.treap()
    Utup = hash(tuple(U.ravel()))
    uc = 0
    g = 0
    h, w = transWt(U,r1,r2)
    v = ((0,0,0,0),(0,0))
    o = astarObj(g,h,None,v,w,Utup)
    Q[o.w] = o
    visited = {Utup:o}
    deleted = set()
    children = {Utup:set()}
    while Q.length > 0:
        w,o = Q.remove_min()
        UC = o.getUC(U)
        if (o.h == 0):
            return o,UC
        g = o.g + 2
        for acbd,ij in ijOptions(UC):
            Ui = TvMul(UC,acbd,ij)
            Utup = hash(tuple(Ui.ravel()))
            if Utup not in visited:
                children[o.U].add(Utup)
                hi,wi = transWt(Ui,r1,r2)
                oi = astarObj(g,hi,o,(acbd,ij),wi,Utup)
                visited[Utup] = oi
                children[Utup] = set()
                Q[oi.w] = oi
            else:
                oi = visited[Utup]
                if oi.g > g:
                    uc +=1
                    oldparent = oi.parent.U
                    # print('oldparent',oldparent,len(children[oldparent]))
                    children[oldparent].remove(Utup)
                    oi.parent = o
                    children[o.U].add(Utup)
                    # print(Utup,'deltaG',deltaG)
                    toUpdate = [oi.U]
                    updated = set()
                    while len(toUpdate) > 0:
                        # print(Utup,'len(toUpdate)',len(toUpdate))
                        Uj = toUpdate.pop()
                        oj = visited[Uj]
                        if Uj in deleted:
                            deleted.remove(Uj)
                        else:
                            Q.root.remove(Q.root, oj.w)
                        # else:
                        #     if Q.search(oj.w) is not False:
                        #         Q.remove(oj.w)
                        oj.updateG(oj.parent.g + 2)
                        Q[oj.w] = oj
                        for Uk in children[Uj]:
                            if Uk not in updated:
                                toUpdate.append(Uk)
        if Q.length > qMax:
            # print('removing Q max',Q.length, bMax)
            for i in range(qMax,Q.length):
                w, oi = Q.remove_max()
                deleted.add(oi.U)
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
## Quantinuum tket: https://docs.quantinuum.com/tket/
## Various optimisation algorithms - best seems to be FullPeepholeOptimise
#########################################################################################################

def tket2opList(circ):
    '''convert tket circuit to opList'''
    opList = []
    for op in circ.get_commands():
        opList.append((str(op.op),[q.index[0] for q in op.qubits]))
    return opList

def csynth_tket(mytext,option='FullPeepholeOptimise'):
    '''tket synthesis - various options'''
    qc = pytket.qasm.circuit_from_qasm_str(mytext)
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