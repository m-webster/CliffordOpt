from .common import *
from .NHow import *
from .CliffordOps import *
import numpy as np
import igraph as ig
import sqlite3
import concurrent.futures
from importlib import resources as impresources
from . import opt
import qiskit, qiskit.circuit, qiskit.qasm2
import os

## required for benchmarking against other methods, but not core functionality
from mqt import qmap, qecc
import stim
import treap
import pyzx
import pytket, pytket.tableau, pytket.passes, pytket.qasm
import rustiq

#########################################################################################################
## Main Synthesis Algorithms
#########################################################################################################

def synth_Sp(U,params):
    '''synthesis - Symplectic matrix for Clifford circuits'''
    return synth_main(U,params)

def synth_GL(U,params):
    '''synthesis - GL matrix for CNOT circuits'''
    ## convert GL to Symplectic matrix and run synth_main
    return synth_main(symCNOT(U),params)

def synth_QC(mytext,params):
    '''synthesis - Clifford circuit in QASM text format'''
    ## make a Qiskit quantum circuit QASM text
    qc = qiskit.QuantumCircuit.from_qasm_str(mytext)
    ## convert to symplectic matrix
    U = ZMat(qiskit.quantum_info.Clifford(qc).symplectic_matrix)
    return synth_main(U,params,qc)

def synth_main(U,params,qc=None):
    '''main synthesis function - calls various optimsation functions using options in params. U is a symplectic matrix, qc is a Qiskit circuit'''
    if qc is None:
        qc = sym2qc(U)
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
        opList = csynth_opt(U,params)

    ## Astar Algorithm
    elif params.method == 'astar':
        opList, UC = synth_astar(U,params)
        opList = mat2SQC(UC) + opList

    ############################################################
    ## CNOT Circuit Methods
    ############################################################

    ## CNOT algorithms from Low-overhead Magic State Circuits with Transversal CNOTs https://arxiv.org/abs/2501.10291
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

    ## PMH Algorithm: Optimal synthesis of linear reversible circuits, https://ieeexplore.ieee.org/abstract/document/10313691
    elif params.method == 'CNOT_Patel':
        opList = CNOT_Patel(U[:m,:n])

    ## Brugiere et al: Gaussian Elimination versus Greedy Methods for the Synthesis of Linear Reversible Circuits, https://doi.org/10.1145/3474226
    elif params.method == 'CNOT_brug':
        opList = CNOTBrug(U[:m,:n])

    # ############################################################
    # ## General Clifford Methods
    # ############################################################

    ## pytket: Sivarajah et al, t|ket⟩: a retargetable compiler for NISQ devices, https://dx.doi.org/10.1088/2058-9565/ab8e92
    elif params.method == 'pytket':
        opList = csynth_tket(qc,params.methodName)

    ## rustiq: Brugiere et a, A graph-state based synthesis framework for Clifford isometries, https://doi.org/10.22331/q-2025-01-14-1589
    elif params.method == 'rustiq':
        opList = csynth_rustiq(U)

    ## pyzx: Kissinger, PyZX: Large Scale Automated Diagrammatic Reasoning, http://dx.doi.org/10.4204/EPTCS.318.14
    elif params.method == 'pyzx':
        opList = csynth_pyzx(qc)

    ## qiskit: greedy algorithm from Clifford Circuit Optimization with Templates and Symbolic Pauli Gates, https://doi.org/10.22331/q-2021-11-16-580
    elif params.method == 'qiskit':
        circ = csynth_qiskit(qc,params.methodName)
        opList = qiskit2opList(circ)

    ## volanto: Minimizing the Number of Two-qubit Gates in Clifford Circuits, https://doi.org/10.1007/978-3-031-26553-2_7
    elif params.method == 'volanto':
        opList, UC = csynth_volanto(U)
        opList = mat2SQC(UC) + opList
    
    ## stim: Gidney, Stim: a fast stabilizer circuit simulator, https://doi.org/10.22331/q-2021-07-06-497
    elif params.method == 'stim':
        opList = csynth_stim(U)
    
    ## if no method specified, just count gates in input circuit
    else:
        opList = qiskit2opList(qc)

    depth = len(opListLayers(opList))
    gateCount = entanglingGateCount(opList)
    procTime = currTime()-sT
    circ = opList2str(opList,ch=" ")
    MWalgs = ['optimal','volanto','greedy','astar','CNOT_optimal','CNOT_gaussian','CNOT_Patel','CNOT_greedy','CNOT_astar','CNOT_depth']
    if params.method in MWalgs:
        check = symTest(U,opList)
    else:
        check = ""
    return n,gateCount,depth,procTime,check,circ


#########################################################################################################
## MW greedy algorithm
#########################################################################################################

def csynth_greedy(A,params):
    '''Decomposition of symplectic matrix A into 2-transvections, SWAP and single-qubit Clifford layers'''
    mode = params.mode
    m,n = symShape(A)
    A = A.copy()
    opList = []
    hix = 1 if params.hv else 0
    h,w = GLHeuristic(A,params) if mode=='GL' else  SpHeuristic(A,params)
    hMin,hLast = None,None
    currWait = 0
    dMax = 10000
    while h > 0.00001:
        gateOpts = GLOptions(A,False) if mode == 'GL' else  SpOptions(A)
        dhMin,BMin = None,None
        for myOp in gateOpts:
            B = applyOp(A,myOp) 
            h,w = GLHeuristic(B,params) if mode=='GL' else  SpHeuristic(B,params)
            hB = (w,h,myOp) if params.hv else (h,w,myOp)
            if params.minDepth:
                d = len(opListLayers(opList + [myOp])) 
                if hLast is not None and hB > hLast:
                    d += dMax
            else:
                d = 0
            dhB = (d,hB)
            if (dhMin is None) or (dhMin > dhB):
                dhMin,BMin = dhB,B
        # print(dhMin)
        hLast = dhMin[1]
        h = hLast[hix]
        opList.append(hLast[-1])
        A = BMin
        if hMin is None or hLast < hMin:
            currWait = 0
            hMin = hLast
        else:
            currWait += 1
        if (params.wMax > 0 and currWait > params.wMax):
            return [],np.arange(n),[]
    opList = opListInv(opList)
    return opList,A

def SpOptions(U):
    '''Transvection gate options for symplectic reduction'''
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

def GLOptions(A,allOpts=False):
    '''CNOT gate options for GL reduction'''
    m,n = symShape(A)
    # return [(i,j) for i in range(n) for j in range(n) if i != j]
    U = A[:m,:n]
    if allOpts:
        return [('CX',(i,j)) for i in range(n) for j in range(n) if i!=j]
    ## dot product of columns with columns - non-zero elements have overlap and so are in the list
    iList,jList = np.nonzero((U.T @ U)) 
    ## exclude those along the diagonal
    return [('CX',(i,j)) for (i,j) in zip(iList,jList) if i != j]

def SpHeuristic(U,params):
    '''calculate heuristics - vector and scalar - for symplectic matrices'''
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
        # h = hr * np.sum(np.log(c1 + c2))/len(c1)
        h = matSum(np.log(c1 + c2))/len(c1)
    else:
        h = (matSum(UR1) + matSum(UR2))/n - 1
    return hr * h, tuple(sorted(c2 * n + c1))

def GLHeuristic(U,params):
    '''calculate heuristics - vector and scalar - for symplectic matrices'''
    m,n = symShape(U)
    if params.hi == 0:
        U = U[:m,:n]
    sA = vecJoin(matColSum(U),matColSum(U.T)) if params.ht else matColSum(U)
    # Ls = len(sA) if params.hl else len(U[0])
    Ls = len(sA)
    h =  matSum(np.log(sA))/Ls if params.hl else (matSum(U)/len(U) - 1)
    return params.hr * h, tuple(sorted(sA))

#########################################################################################################
## MW Astar algorithm 
#########################################################################################################

def synth_astar(U,params):
    '''Astar using treap to manage size of priority queue'''
    mode,qMax = params.mode,params.qMax
    Q = treap.treap()
    Utup = hash(tuple(U.ravel()))
    currId = 0
    visited = {Utup: currId}
    g = 0
    h, w = GLHeuristic(U,params) if mode=='GL' else SpHeuristic(U,params)
    Aid = -1
    op = ((0,0,0,0),(0,0),0,0)
    DB = [(Aid,op,g,h)]
    Q[(g+h,g,h,w,op)] = currId
    while Q.length > 0:
        s,Aid = Q.remove_min()
        _,_,g,h = DB[Aid]
        opList = getOpListRec(Aid,DB)
        A = applyOpList(opList,U)
        if (h < 0.000001):
            return opListInv(opList),A
        myOpts = GLOptions(A) if mode=='GL' else SpOptions(A)
        for myOp in myOpts:
            Ui = applyOp(A,myOp)
            Utup = hash(tuple(Ui.ravel()))
            if Utup not in visited:
                h, w = GLHeuristic(Ui,params) if mode=='GL' else SpHeuristic(Ui,params)
                currId += 1
                visited[Utup] = currId
                gi = len(opListLayers(opList + [myOp])) if params.minDepth else g + 1
                DB.append((Aid,myOp,gi,h))
                Q[(gi+h,gi,h,w,myOp)] = currId
        if Q.length > qMax:
            for i in range(qMax,Q.length):
                Q.remove_max()
    return None

def getOpListRec(A,visited):
    '''Recursive method to get opList from tree structure'''
    p,myOp = visited[A][0],visited[A][1]
    if p < 0:
        return []
    parentOps = getOpListRec(p,visited)
    return parentOps + [myOp]

def sym2tuple(U,mode):
    '''flatten U to a tuple - in the case of GL, just take U_XX component'''
    if mode == 'GL':
        m,n = symShape(U)
        U = U[:m,:n]
    return tuple(np.reshape(U,-1))

#########################################################################################################
## Optimal Synthesis 
#########################################################################################################

def csynth_opt(A,params):
    '''optimal synthesis using stored circuit implementations'''
    mode, minDepth = params.mode,params.minDepth
    m,n = symShape(A)
    AList,hA,tA,iA = matOpts(A)
    ACerts = [MatCanonize(U,mode) for U in AList]
    ACert = min(C[1] for C in ACerts)
    DBName = getDBName(n,mode,minDepth)
    cnx,cur = dbConnect(DBName)
    tableName = 'BData'
    cmd = f'select BId,hex(B) from {tableName} where BCert = ?'
    cur.execute(cmd,[ACert])
    myData = cur.fetchone()
    BId,B = myData
    opList = DB2OpList(cur,tableName,BId)
    B = bytes2sym(B,n)
    Bix,BCert = MatCanonize(B,mode)
    i = [C[1] for C in ACerts].index(BCert)
    trans,inv,Aix = tA[i],iA[i],ACerts[i][0]
    ix = ZMat(Bix[ixRev(Aix)])
    r, c = m, n
    if mode != 'GL':
        r,c = 3*r,3*c
    perms = [ix[:r], ix[r:] - r]
    return DB2GL2(opList,trans,inv,perms) if mode=='GL' else DB2sym(opList,trans,inv,perms)

def DB2GL2(opList,trans,inv,perms):
    '''optimal synthesis - extract circuit for GL matrices'''
    opList = [('QPerm',ixRev(perms[0]))] + opList + [('QPerm',(perms[1]))]
    if trans:
        opList = opListT(opList)
    if inv:
        opList = opListInv(opList)
    opList = QPerm2Front(opList)
    return opList

def DB2sym(opList,trans,inv,ixList):
    '''optimal synthesis - extract circuit for symplectic matrices'''
    n = len(ixList[1]) // 3
    E = EMat(n)
    EInv = EInvMat(n)
    perms,cliffs = [],[]
    for i in range(2):
        ix = ixList[i]
        UC = matMul(E[:,ix], EInv,2)[:2*n,:2*n]
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


#########################################################################################################
## CNOT Synthesis from from Low-overhead Magic State Circuits with Transversal CNOTs https://arxiv.org/abs/2501.10291
#########################################################################################################

def CNOT_greedy(A,verbose=False):
    '''optimisation of CNOT-count from Low-overhead Magic State Circuits with Transversal CNOTs https://arxiv.org/abs/2501.10291'''
    A = A.T.copy()
    m,n = A.shape
    done = (np.sum(A) == len(A))
    opList = []
    stepCount = 1
    if verbose:
        print(ZMatPrint(A))
        print('Sorted Row/Col Weights:',matWt(A))
    while not done:
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
    '''Optimisation of CNOT circuit depth from Low-overhead Magic State Circuits with Transversal CNOTs https://arxiv.org/abs/2501.10291'''
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

def matWt(A):
    '''sorted weights of columns and rows, returned as tuple for sorting'''
    A = ZMat(A)
    sA = tuple(sorted(vecJoin(matColSum(A),matColSum(A.T))))
    return sA

#########################################################################################################
## CNOT Synthesis -  Gaussian Elimination
#########################################################################################################

def CNOT_GaussianElim(A):
    '''Gaussian Elimination CNOT Circuit Synthesis - reduce to I'''
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
    '''Patel CNOT synthesis - asymptotically optimal'''
    A = A.T.copy()
    n = len(A)
    m = max(int(np.round(np.log2(n)/2)),2)
    A, opList1 = CNOT_Synth_lwr(A, m,useSWAP)
    At, opList2 = CNOT_Synth_lwr(A.T, m,useSWAP)
    opList = (opList1 + opListT(opList2))
    opList = opListInv(opList)
    opList = QPerm2Front(opList)
    return opList

def CNOT_Synth_lwr(A, m,useSWAP=True):
    '''helper function for CNOT_Patel - process lower triangular matrix'''
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
## Volanto Transvection algorithm 
#########################################################################################################

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


##############################################################################
## Helper Functions - Transvection decomposition of Symplectic Matrix
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
## Qiskit Synthesis
#########################################################################################################

def csynth_qiskit(qc,method='greedy'):
    '''qiskit synthesis - various methods
    input is qiskit circuit qc''' 
    qc = qiskit.quantum_info.Clifford(qc)
    if method == 'ag':
        return qiskit.synthesis.synth_clifford_ag(qc)
    elif method == 'layers':
        return qiskit.synthesis.synth_clifford_layers(qc)
    else:
        return qiskit.synthesis.synth_clifford_greedy(qc)

def qiskit2opList(circ):
    '''convert qiskit circuit to opList'''
    opList = []
    for op in circ.data:
        opName = op.operation.name.upper()
        qList = [q._index for q in op.qubits]
        opList.append((opName, qList)) 
    return opList

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

def csynth_pyzx(qc):
    mytext = qc2qasm(qc)
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

def csynth_tket(qc,option='FullPeepholeOptimise',nReps=10):
    '''tket synthesis - various options'''
    mytext = qc2qasm(qc)
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

def tket2opList(circ):
    '''convert tket circuit to opList'''
    opList = []
    for op in circ.get_commands():
        opList.append((str(op.op),[q.index[0] for q in op.qubits]))
    return opList

#########################################################################################################
## Stim
#########################################################################################################

def csynth_stim(U):
    '''Stim synthesis - not optimised for 2-qubit gate count'''
    submethods = ['elimination','graph_state']
    xx,xz,zx,zz = sym2components(U)
    T = stim.Tableau.from_numpy(x2x=xx,x2z=xz,z2x=zx,z2z=zz)
    qc = T.to_circuit(method=submethods[0])
    return stim2opList(qc)

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

def sym2components(U):
    '''Split Symplectic matrix U into components XX,XZ,ZX,ZZ for use in stim'''
    n = len(U)//2
    xx=np.array(U[:n,:n],dtype=bool)
    xz=np.array(U[:n,n:],dtype=bool)
    zx=np.array(U[n:,:n],dtype=bool)
    zz=np.array(U[n:,n:],dtype=bool)
    return xx,xz,zx,zz

##########################################################
## rustiq
##########################################################

def csynth_rustiq(U,iter=10):
    '''Synthesize using rustiq package'''
    stabilisers = sym2PauliStr(U)
    return rustiq.clifford_synthesis(stabilisers,rustiq.Metric.COUNT, syndrome_iter=iter)

def sym2PauliStr(U):
    '''convert symplectic matrix to Pauli strings for rustiq'''
    m,n = symShape(U)
    pauli_list = ['I','X','Z','Y']
    Uint = U[:,:n] + 2*U[:,n:] 
    return ["".join([pauli_list[a] for a in myRow]) for myRow in Uint]

#########################################################################################################
## Brugiere greedy algorithm
#########################################################################################################

def CNOTBrug(A,LUmethod='MinCost'):
    '''Brugiere CNOT synthesis using various methods to form lower triangular matrix form'''
    if LUmethod=='MinCost':
        B, OpL,OpR = LUDecompMinCost(A)
    elif LUmethod=='Sparse':
        B, OpL,OpR = LUDecompSparse(A)
    elif LUmethod=='Std':
        B, OpL,OpR = LUDecompStd(A)
    C, OpC = BrugLower(B)
    opList =  opListInv(OpR + OpC + OpL)
    return QPerm2Front(opList)

def LUDecompMinCost(A):
    '''Reduce to lower triangular form - minCost'''
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
    '''Reduce to lower triangular form - sparse'''
    preOps,postOps = [],[]
    n = len(A)
    for k in range(n-1,-1,-1):
        iList,jList = np.nonzero(A[:k+1,:k+1])
        BMin = None
        ij = None
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
    '''Reduce to lower triangular form - standard'''
    preOps,postOps = [],[]
    n = len(A)
    for k in range(n-1,-1,-1):
        iList,jList = np.nonzero(A[:k+1,:k+1])
        i,j = iList[-1],jList[-1]
        A,ixL,CXList,ixR = LU(A,i,j,k)
        preOps = [('QPerm',tuple(ixL))] + preOps
        postOps = postOps + [('QPerm',tuple(ixR))] + CXList
    return A, preOps, postOps

def LU(A,i,j,k):
    '''Helper function - Reduce to lower triangular form'''
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
    for l in range(k):
        if B[k,l] == 1:
            B[:,l] ^= B[:,k]
            CXList.append(('CX',(k,l)))
    return B,ixL,CXList,ixR

def USum(A):
    '''sum of upper triangular entries - termination condition for LU methods'''
    A = AUT(A,True)
    return matSum(A)

def AUT(A,exDiag=False):
    '''upper triangular entries'''
    temp = ZMatZeros(A.shape)
    b = 1 if exDiag else 0
    for i in range(len(A)-b):
        temp[i,i+b:] = A[i,i+b:]
    return temp

def BrugLower(A):
    '''reduce lower triangular matrix to identity'''
    m,n = A.shape
    opList = []
    A = A.copy()
    while not USum(A) == 0:
        i,j = bin2Set(SelectRowOperation(A))
        opList.append(('CX',(i,j)))
        A[:,j] ^= A[:,i]
    return A, opList

def SelectRowOperation(A):
    '''choose row op for BrugLower'''
    m,n = A.shape
    j = 0
    S = ZMat([1]*n)
    while matSum(S) > 2:      
        a = A[j]  
        S0 = (1 ^ a) & S
        S1 = a & S
        S = S0 if matSum(S1) < 2 else S1
        j += 1
    return S

##########################################################
## Utilities for main Synth Algorithms
##########################################################

def sym2qc(U):
    '''Convert symplectic matrix U to a qiskit circuit object'''
    return qiskit.quantum_info.Clifford(U).to_circuit()

def qc2qasm(qc):
    '''convert qiskit circuit object to qasm 2 string'''
    return qiskit.qasm2.dumps(qc)


#################################################################################
## opList Manipulations
#################################################################################

SQC_tostr = {'1001':'I', '0110':'H','1101':'S','1011':'HSH','1110':'HS','0111':'SH'}
SQC_fromstr = {v : np.reshape([int(i) for i in k],(2,2)) for k,v in SQC_tostr.items()}

def opList2sym(opList,n):
    '''convert opList to 2n x 2n binary symplectic matrix '''
    return applyOpList(opList,ZMatI(2*n),True)

def symTest(U,opList):
    '''check that opList circuit is equivalent to symplectic matrix U'''
    m,n = symShape(U)
    U2 = opList2sym(opList,n)
    return binMatEq(U,U2)

def CNOT2opList(ix,opList):
    '''convert output of CNOT algorithm to an opList'''
    return [('QPerm',ix)] + [('CX',[i,j]) for (i,j) in opList]

def opListLayers(opList):
    '''split opList into layers to calculate circuit depth'''
    # opList = SQC2front(opList)
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

def applyOpList(opList,A,update=False):
    '''apply list of operations'''
    if type(opList) == tuple:
        opList = [opList]
    if not update:
        A = A.copy()
    for myOp in opList:
        A = applyOp(A,myOp,True)
    return A

def applyOp(U,myOp,update=False):
    '''apply op to U - update U if update=True'''
    m,n = symShape(U)
    opType,qList = myOp
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
    elif isSQC(opType):
        U = applySQC(U,opType,qList)
    return U

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
    '''check if op is entangling'''
    ## any gate actingh on more than one qubit which is not a SWAP or QPerm
    return len(qList) > 1 and opName != 'SWAP' and opName != "QPerm"

def getSupp(opList):
    '''Support of operators in opList'''
    sList = set()
    for opType,qList in opList:
        sList.update(qList)
    return sList

def str2opList(mystr):
    '''convert string to opList'''
    mystr = mystr.split()
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

####################################################
## Qubit Permutations
####################################################

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

def isIdPerm(ix):
    '''return true if ix is an identity permutation'''
    return nonDecreasing(ix)

####################################################
## Single-qubit Clifford operators SQC
####################################################

def isSQC(opType):
    '''check if opType is a single-qubit Clifford'''
    global SQC_fromstr
    return opType in SQC_fromstr

def applySQC(U,opType,qList):
    '''apply single-qubit Clifford to U'''
    m,n = symShape(U)
    q = qList[0]
    A = str2SQC(opType)
    Ui = matMul(U[:,[q,q+n]],A,2)
    U[:,[q,q+n]] = Ui
    return U

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

def CList2opList(UC):
    '''convert list of 2x2 SQC matrices to opList'''
    temp = []
    ## dict for single-qubit Cliffords
    for i in range(len(UC)):
        c =  SQC2str(UC[i])
        ## don't add single-qubit identity operators
        if c != 'I':
            temp.append((c,[i]))
    return temp

def SQC2str(A):
    '''convert a 2x2 single qubit Clifford matrices to opType'''
    global SQC_tostr
    return SQC_tostr[ZMat2str(A.ravel())] 

def str2SQC(mystr):
    '''convert string to single-qubit Clifford opType'''
    global SQC_fromstr
    return SQC_fromstr[mystr]

def mat2SQC(UC):
    '''convert matrix UC to opList representing qubit permutation and list of single-qubit cliffords'''
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

####################################################
## Transvections
####################################################

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

def Pauli2bin(mystr):
    '''convert Pauli string to tupe representing x and z components'''
    pauliX = {'I':0,'X':1,'Y':1,'Z':0}
    pauliZ = {'I':0,'X':0,'Y':1,'Z':1}
    return tuple([pauliX[a] for a in mystr] + [pauliZ[a] for a in mystr])

def isTv2(opType):
    '''check if opType is 2-qubit transvection'''
    return (typeName(opType) == 'tuple') and (len(opType) == 4)

def TvTransp(opType):
    '''Transpose of transvection  swap X and Z components'''
    m = len(opType)//2
    return opType[m:] + opType[:m]

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

def GLRand(rng,n):
    '''generate random nxn invertible matrix'''
    xList = GLRandVec(rng,n)
    return vec2GL(xList)

def matOpts(A0):
    '''yield list of distinct A0, transp, inv, inv transp'''
    AInv = symInverse(A0)
    AList = []
    hA = set()
    tA = []
    iA = [] 
    A = A0
    for i in range(2):
        for t in range(2):
            h = symHash(A)
            if h not in hA:
                AList.append(A)
                hA.add(h)
                tA.append(t)
                iA.append(i)
            A = A.T
        A = AInv
    return AList,hA,tA,iA

def symHash(U):
    '''hash of U expressed as a 1D tuple of 0/1'''
    return hash(tuple(np.ravel(U)))

########################################################
## igraph Bliss canonization
########################################################

def MatCanonize(U,mode):
    '''Canonize the matrix U using graph isomorphism method'''
    m,n = symShape(U)
    ## Edge tuple list method
    if mode == 'GL':
        EList = binMat2Graph(U[:m,:n])
        r, c = m,n
    else:
        EList = sym2Graph(U)
        r, c = m*3,n*3
    G = ig.Graph(r + c,EList)
    ## one colour for rows, one for edges
    colouring = [0] * r + [1] * c
    ## get canonical labelling and convert to Graph6 format
    ix = G.canonical_permutation(color=colouring)
    G = G.permute_vertices(ix)
    B = np.array(G.get_adjacency(),np.int8)
    cert = adj2Graph6(B)
    return ixRev(ZMat(ix)),cert

def minCert(B0,mode):
    '''return minimal certificate for B0, transp, inv and inv transp'''
    BList,hB,_,_ = matOpts(B0)
    BCert = None
    for B in BList:
        ix,cert = MatCanonize(B,mode)
        if BCert is None or cert < BCert:
            BCert = cert
    return BCert

def binMat2Graph(U):
    '''Transform binary matrix U to list of tuples indicating which rows are connected to columns'''
    m,n = U.shape
    iList,jList = np.nonzero(U)
    if m > 0:
        jList+=m
    return list(zip(iList,jList))

mxnDict = dict()

def cliffAdjmxn(m,n):
    '''Create edges on rows/cols to restrict to single-qubit Cliffords and qubit swaps'''
    # save to global variable to reduce processing time
    global mxnDict
    if (m,n) not in mxnDict:
        ## save to global variable
        mxnDict[(m,n)] = cliffAdj(m) + cliffAdj(n,m)
    return mxnDict[(m,n)]

def cliffAdj(n,m=0):
    '''Create edges on rows/cols to restrict to single-qubit Cliffords and qubit swaps'''
    ## 3 times to reflect X, Z and X+Z components
    c = 3*n
    ## m is an offset - for creating the col edges
    r = 3*m
    return [ (i+r, ((i + j * n) %  c) + r) for j in range(1,3) for i in range(c) ]

def sym2Graph(U):
    '''Create a graph from a symplectic matrix U'''
    m,n = symShape(U)
    ## create U_3n matrix with X, Z and X+Z components for rows and cols
    U3 = ZMatZeros((m*3,n*3))
    U3[:2*m,:2*n] = U
    U3[:2*m,2*n:] = U[:,:n] ^ U[:,n:]
    U3[2*m:,:] = U3[:m,:] ^ U3[m:2*m,:]
    ## add edges to restrict to single-qubit Cliffords and SWAP for rows/cols
    return cliffAdjmxn(m,n) +  binMat2Graph(U3)

@nb.njit(nb.types.unicode_type(nb.int8[:,:]))
def adj2Graph6(A):
    '''Convert adjacency matrix A to graph6 format'''
    m,n = A.shape
    temp = 0
    count = 0
    bin_list = [chr(n + 63)] 
    for i in range(1,n):
        for j in range(i):
            temp = temp << 1
            temp ^= A[i,j]
            count += 1
            if count == 6:
                bin_list.append(chr(temp + 63))
                temp = 0
                count = 0
    if count > 0:
        bin_list.append(chr((temp << (6-count)) + 63))
    return "".join(bin_list)

#################################################
## Generate Optimal Circuit Database
#################################################

def ResumeDBGen(n,mode='GL',nWorkers=8,minDepth=False):
    '''Generate DB for optimal synthesis - can be resumed in the case of interruption'''
    startTimer()
    sT = currTime()
    DBName =   getDBName(n,mode,minDepth)
    tableName = 'BData'
    print(f'Calculating database {DBName}')
    myFile = f'{DBName}.db'
    if not os.path.exists(myFile):
        ## Start with identity matrix
        B = ZMatI(2*n)
        ix, BCert = MatCanonize(B,mode)
        certLen = len(BCert)
        ## Set up new table
        cnx,cur = dbConnect(DBName)
        setupTable(cnx,tableName,certLen,n)
        ## insert identity matrix as only d=0 entry
        d,AId,opList,transp,inv = 0,-1,"",0,0
        myData = (sym2bytestr(B),BCert,AId,opList,transp,inv,d)
        insertRows(cnx,[myData])
        cnx.commit()
        BId,SId = 1,1
    else:
        cnx,cur = dbConnect(DBName)
        cur.execute(f'select d,max(BId),length(BCert) from {tableName} group by d;')
        BDict = dict()
        for d,BId,certLen in cur.fetchall():
            BDict[d] = BId
        if len(BDict) > 0:
            d = max(BDict.keys())
            BId = BDict[d]
            SId = 1 if d == 0 else BDict[d-1]+1
        else:
            d,SId,BId = -1,1,-1
    print(f'd:{d} SId:{SId} BId:{BId}')
    while SId <= BId:
        d += 1
        BRange = BId - SId + 1
        print(f'Calculating level {d} circuits from {BRange} level {d-1} circuits {elapsedTime():.3f}')
        if BRange > 10:
            steps0,steps1 = getSteps(SId,BId,nWorkers)
            print(f'Running {len(steps0)} threads')
            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                threadFuture = {executor.submit(optDBstep,n,mode,minDepth,d,steps0[i],steps1[i],i,certLen): i for i in range(len(steps1))}
                for future in concurrent.futures.as_completed(threadFuture):
                    i = threadFuture[future]
                    ## merge once thread completed
                    mergeData(cnx,n,mode,minDepth,i)
            ## just in case of threads crashing, make sure we merge data at the end also
            print(f'All level {d} threads completed - merging')
            for i in range(len(steps1)):
                mergeData(cnx,n,mode,minDepth,i)
        else:
            optDBstep(n,mode,minDepth,d,SId,BId,None,None)
        ## starting index for next iteration = BId + 1
        SId = BId + 1
        ## BId for next iteration is row with highest Id
        cmd = f'select max(BId) from {tableName};'
        cur.execute(cmd)
        BId = int(cur.fetchone()[0])
    cnx.close()
    print(f'Run time {currTime() - sT:.3f}')
    return BId

def getSteps(SId,BId,nWorkers):
    '''split the range [SId..BId] into up to blocks to be processed by nWorkers'''
    BRange = BId-SId + 1
    stepSize = (BRange-1)//nWorkers + 1
    nSteps = (BRange-1)//stepSize + 1
    steps0 =  {min(BId,SId + i * stepSize) for i in range(nSteps)}
    steps1 = {min(BId,SId + (i+1) * stepSize-1) for i in range(nSteps)}
    steps0 = sorted(steps0)
    steps1 = sorted(steps1)
    return steps0,steps1

def optDBstep(n,mode,minDepth,d,SId,BId,iter,certLen):
    '''form new matrices from SId <= id <= BId by applying all possible gates'''
    maxLen = 50000
    tableName = 'BData'
    DBName = getDBName(n, mode, minDepth)
    if iter is not None:
        DBiter = DBName + f'_{iter:03}'
        if not os.path.exists(f'{DBiter}.db'):
            cnx,cur = dbConnect(DBiter)
            setupTable(cur,tableName,certLen,n)
        else:
            cnx,cur = dbConnect(DBiter)
            cur.execute(f'select max(AId) from {tableName};')
            res = cur.fetchone()
            if res is not None:
                SId = res[0]
    print(f'd:{d} iter:{iter} SId:{SId},BId:{BId}')
    ## save possible moves  - note that that these are lists of ops
    if not minDepth:
        GateOpts = [[op] for op in opMoves(n,mode)]
    cnx,cur = dbConnect(DBName)
    cmd = f'select `BId`,hex(`B`),`op` from `{tableName}` where `BId` >= ? and `BId` <= ?'
    cur.execute(cmd,(SId,BId))
    AList = cur.fetchall()
    if iter is not None:
        cnx,cur = dbConnect(DBiter)
    BData = []
    for myrow in AList:
        AId = myrow[0]
        A0 = bytes2sym(myrow[1],n)
        AList,hA,tA,iA = matOpts(A0)
        ## refined method for depth-opt circuits
        if minDepth:
            opListA = str2opList(myrow[2])
            GateOpts = opMovesD(n,opListA,mode)
        for (A,Atrans,Ainv) in zip(AList,tA,iA):
            for opList in GateOpts:
                B0 = applyOpList(opList,A)
                BCert = minCert(B0,mode)
                BData.append((sym2bytestr(B0),BCert,AId,opList2str(opList," "),Atrans,Ainv,d))
        if len(BData) > maxLen:
            insertRows(cnx,BData)
            BData = []
    if len(BData) > 0:
        insertRows(cnx,BData)
    cnx.close()
    return 1

## SQLite3 Database

def dbConnect(DBName):
    '''Connect to DBName'''
    cnx = sqlite3.connect(f'{DBName}.db')
    cur = cnx.cursor()
    return cnx,cur

def getDBName(n, mode, minDepth):
    '''Get standard DB name - more complicated as we need to connect to package resources'''
    with impresources.path('cliffordopt', 'opt') as myPath:
        D = "D" if minDepth else ""
        return f'{myPath}/{mode}{D}_{n}'
    return None

def insertRows(cnx,dataList):
    '''insert multiple rows into optimal database'''
    tableName = 'BData'
    fieldNames = ['B','BCert','AId','op','transp','inv','d']
    fn = ", ".join(fieldNames)
    vals = ", ".join(["?"] * len(fieldNames))
    cmd = f'insert or ignore into `{tableName}` ({fn}) VALUES ({vals});'
    cnx.executemany(cmd,dataList)
    cnx.commit()
    return 0

def setupTable(cur,tableName,certLen,n):
    '''set up table for optimal database'''
    matLen = (4 * n * n - 1) // 8 + 1
    cmd = f'CREATE TABLE `{tableName}` (BId integer not null primary key, B BINARY({matLen}) , BCert CHAR ({certLen}) UNIQUE, AId int, op text, transp bool, inv bool, d tinyint );'
    cur.execute(cmd)


def mergeData(cnx,n,mode,minDepth,ixThread):
    '''merge data from helper threads back into main database'''
    dbName =   f'{getDBName(n,mode,minDepth)}_{ixThread:03}'
    myFile = f'{dbName}.db'
    if os.path.exists(myFile):
        print(f'merging {ixThread}')
        tableName = "BData"
        fieldNames = ["B","BCert","AId","op","transp","inv","d"]
        fn = ", ".join(fieldNames)
        fnA = ", ".join([f"A.{s}" for s in fieldNames])
        cnx.execute(f"ATTACH '{dbName}.db' as dbi")
        cmd = f'insert into {tableName} ({fn}) select {fnA} from dbi.{tableName} A where not exists (select A.BCert from {tableName} as B where A.BCert = B.BCert);'
        cnx.execute(cmd)
        cnx.commit()
        cnx.execute("detach database dbi")
        DBclear(dbName)

def DBclear(DBName):
    '''Clear helper optimal database file'''
    myFile = f'{DBName}.db'
    if os.path.exists(myFile):
        os.remove(myFile)


## Options for 2-qubit gates


def DepthPart(qList,sList):
    '''Return lists of tuples of qubit indices in qList which increase depth of operator with support sList'''
    temp = [[]]
    sL = len(qList)
    for i in range(sL-1):
        qi = qList[i] in sList
        for j in range(i+1,sL):
            p = (qList[i],qList[j])
            ## make sure we have overlap with sList - ensures we are increasing circuit depth
            if qi or (qList[j] in sList):
                qList1 = [qList[k] for k in range(i+1,sL) if k != j]
                temp.extend([[p] + a2 for a2 in DepthPart(qList1,sList)])
    return temp

def opMovesD(n,opList,mode='GL'):
    '''get possible moves - either CNOT (GL) or 2-qubit transvections'''
    qList = np.arange(n)
    if len(opList) > 0:
        sList = getSupp(opList)
    else:
        sList = set(qList)
    temp = []
    opTypes = [0,1] if mode=='GL' else [(a % 2,b%2,a//2,b//2) for a in range(1,4) for b in range(1,4)] 
    pList = DepthPart(qList,sList)
    for ijList in pList:
        if len(ijList) > 0:
            ijL = len(ijList)
            for opTypeList in iter.product(opTypes,repeat=len(ijList)):
                if mode == 'GL':
                    ijCopy = [ijList[i] if opTypeList[i] == 1 else tuple(reversed(ijList[i])) for i in range(ijL)]
                    ops = list(zip(['CX']* ijL,ijCopy))
                else:
                    ops = list(zip(opTypeList,ijList))
                temp.append(ops)
    sortable = sorted([(len(temp[i]),i) for i in range(len(temp))])
    temp = [temp[op[1]] for op in sortable]
    return temp

def opMoves(n,mode='GL'):
    '''get possible moves - either CNOT (GL) or 2-qubit transvections'''
    temp = []
    vList = {(a % 2,b%2,a//2,b//2) for a in range(1,4) for b in range(1,4)} 
    for i in range(n-1):
        for j in range(i+1,n):
            ij = [i,j]
            if mode == 'GL':
                for k in range(2):
                    temp.append(('CX',tuple(ij)))
                    ij.reverse()
            else:
                for acbd in vList:
                    temp.append((acbd,(i,j)))
    return temp

def retrieveRow(cur,tableName,BId):
    cmd = f'select * from `{tableName}` where `BId`=?;'
    cur.execute(cmd,[BId])
    return cur.fetchone()

def DB2OpList(cur,tableName,BId):
    '''Recursive method to get opList from tree structure'''
    data = retrieveRow(cur,tableName,BId)
    # print(data)
    if data is None:
        return None
    BId, B, BCert,AId,opList,transp,inv,d = data
    if AId < 1:
        return [] 
    ## convert string to opList
    opList = str2opList(opList)
    parentOps = DB2OpList(cur,tableName,AId)
    if inv:
        parentOps = opListInv(parentOps)
    if transp:
        parentOps = opListT(parentOps)
    return parentOps + opList

## Convert between symplectic matrix and bytes

def bytes2sym(A,n):
    matLen = 4 * n * n
    # print((A))
    temp = [int(a,16) for a in A]
    temp = "".join([f'{a:04b}' for a in temp])
    temp = [int(a) for a in temp[:matLen]]
    return np.array(np.reshape(temp,(2*n,2*n)),dtype=np.int8)

def sym2bytestr(B):
    BList = list(np.ravel(B))
    temp = ['0x']
    temp = []
    for i in range(len(BList)//4):
        a = 0
        for j in range(4):
            a = a << 1
            a ^= BList[4*i + j]
        temp.append(f'{a:x}')
    if len(temp) % 2 != 0:
        temp.append('0')
    return bytes.fromhex("".join(temp))



#######################################################
## Analyse optimal databases
#######################################################

def correlDB(mode,n,minDepth=False):
    '''Correlation between depth/gate count and H metrics'''
    DBName = getDBName(n,mode,minDepth)
    params = paramObj()
    params.mode = mode
    params.hi = 1
    params.ht = 1
    params.hr = 1
    tableName = 'BData'
    cnx,cur = dbConnect(DBName)
    cmd = f'DROP TABLE IF EXISTS `corr`'
    cur.execute(cmd)
    cmd = f'CREATE TABLE IF NOT EXISTS `corr` (d integer, Hs float, Hl float);'
    cur.execute(cmd)
    cmd = f'select hex(B),d from {tableName}'
    cur.execute(cmd)
    for A,d in cur.fetchall():
        A = bytes2sym(A,n)
        A = ZMat(A)
        params.hl = 0
        Hs,hv = GLHeuristic(A,params) if mode=='GL' else SpHeuristic(A,params)
        params.hl = 1
        Hl,hv = GLHeuristic(A,params) if mode=='GL' else SpHeuristic(A,params)
        myrow = [d,Hs,Hl]
        cmd = f'insert into corr values (?,?,?)'
        cnx.execute(cmd,myrow)
        cnx.commit()
    lCorr = colCorr(cnx,'Hl','d')
    sCorr = colCorr(cnx,'Hs','d')
    cnx.close()
    return lCorr,sCorr

def colCorr(cnx,xCol,yCol):
    '''calculate correlation between xCol and yCol'''
    cur = cnx.cursor()
    cmd = f'select count({xCol}) from corr'
    cur.execute(cmd)
    N = cur.fetchone()[0]
    print('N',N)
    cmd = f'select sum({xCol}),sum({yCol}),sum({xCol} * {xCol}),sum({yCol} * {yCol}),sum({xCol} * {yCol}) from corr'
    print(cmd)
    cur.execute(cmd)
    Sx,Sy,Sxx,Syy,Sxy = cur.fetchone()
    Cxx = (N * Sxx - Sx * Sx)
    Cxy = (N * Sxy - Sx * Sy)
    Cyy = (N * Syy - Sy * Sy)
    r2 = Cxy * Cxy / (Cxx * Cyy)
    b = Cxy/Cxx
    a = (Sy - b * Sx)/N
    return a,b,r2 ** 0.5

def analyseOptDB(n,mode,minDepth=False):
    '''analyse optimal database'''
    DBName = getDBName(n,mode,minDepth)
    tableName = 'BData'
    cnx,cur = dbConnect(DBName)
    cmd = f'select d,count(BId) from {tableName} group by d'
    cur.execute(cmd)
    temp = list(cur.fetchall())
    cnx.close()
    return temp

def extractRandom(n,mode,minDepth=False,target=10):
    '''extract random matrices from database'''
    DBName = getDBName(n,mode,minDepth)
    tableName = 'BData'
    cnx,cur = dbConnect(DBName)
    cmd = f'select d,min(BId),max(BId) from {tableName} group by d'
    cur.execute(cmd)
    AList = []
    dList = []
    for (d,SId,BId) in cur.fetchall():
        mySize = min(BId-SId+1,target)
        myIds = np.random.choice(range(SId,BId+1),size=mySize,replace=False)
        cmd = f'select hex(B) from {tableName} where BId in ({", ".join(map(str,myIds))})'
        cur.execute(cmd)
        BList = [bytes2sym(myrow[0],n) for myrow in cur.fetchall()]
        if mode == 'GL':
            BList = [applyRandPerm(B) for B in BList]
        else:
            BList = [applyRandPermSQC(B) for B in BList]
        BList = [ZMat2str(B.ravel()) for B in BList]
        AList.extend(BList)
        dList.extend([d]*len(BList))
    return AList,dList

def applyRandPerm(A):
    '''apply random permutation to A'''
    m,n = A.shape
    ixL = np.random.permutation(n)
    ixR = np.random.permutation(n)
    return A[ixL][:,ixR]

def applyRandPermSQC(A):
    '''apply a random perm+SQC to A'''
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