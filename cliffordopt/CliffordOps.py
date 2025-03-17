import numpy as np
from .NHow import *


##########################################################################
## Basics of Symplectic Matrices
##########################################################################

def OmegaMat(n):
    '''Omega is the Symplectic form on 2n qubits
    Also Hadamard on all n qubits'''
    On = ZMatZeros((2*n,2*n))
    In = ZMatI(n)
    On[:n,n:] = In
    On[n:,:n] = In
    return On

def permMat(perm):
    '''take permutation vector and convert to perm matrix'''
    n = len(perm)
    temp = np.zeros((n,n),dtype=int)
    for i in range(n):
        temp[perm[i],i] = 1
    return temp

def isSymplectic(S):
    '''check if matrix S is binary symplectic'''
    m,n = symShape(S)
    Om = OmegaMat(n)
    return 0 == np.sum(matMul(S.T, matMul(Om,S,2),2) ^ Om)
    return 0 == np.sum(np.mod(S.T @ Om @ S - Om, 2))

# def symInverse(S):
#     '''Inverse of Sympletic matrix S'''
#     n = len(S)//2
#     On = OmegaMat(n)
#     return matMul(On,matMul(S.T,On,2),2)

def symHad(S):
    '''Conjugate S by Had - i.e. swap X and Z components'''
    m,n = symShape(S)
    ix = np.arange(n)
    ix = vecJoin(n+ix,ix)
    return S[:,ix]

def binMatEq(A,B):
    return (matSum(A ^ B) == 0)

def symInverse(S):
    '''Inverse of Sympletic matrix S = On S.T On'''
    m,n = symShape(S)
    ix = np.arange(n)
    ix = vecJoin(n+ix,ix)
    temp = S.T[ix][:,ix]
    # if not binMatEq(matMul(S,temp,2),ZMatI(2*n)):
    #    print(func_name(),'error')
    return temp

def symShape(U):
    m,n = U.shape
    return m//2, n//2

def transvection(v):
    '''Construct symplectic matrix Tv correponding to transvection of vector v'''
    n = len(v)//2
    OvT = ZMatZeros((2*n,1))
    OvT[:n,0] = v[n:]
    OvT[n:,0] = v[:n]
    Tv = matMul(OvT,v,2)
    for i in range(2*n):
        Tv[i,i] ^= 1
    return Tv


# def matWt(A):
#     # sorted weights of columns and rows, returned as tuple for sorting
#     sA = tuple(sorted(np.hstack([np.sum(A,axis=-1),np.sum(A,axis=0)]),reverse=False))
#     return sA

def permMat2ix(P):
    ## Check if this is a permutation matrix
    # print(func_name(),P)
    P = ZMat(P)
    if np.any(matColSum(P.T)!=1):
        return False
    m,n = P.shape
    ix = ZMatZeros(m)
    A = np.nonzero(P)
    for i in range(len(A[1])):
        ix[i] = A[1][i]
    return ix

def binMatInv(A):
    H, U = getHU(A,2)
    return U

## Symplectic form of well-known Clifford operators 

def SymSWAP(ix):
    '''Symplectic matrix corresponding to qubit permutation ix'''
    n = len(ix)
    M = permMat(ix).T
    SOp = ZMatZeros((2*n,2*n))
    SOp[:n,:n] = M
    SOp[n:,n:] = M
    return SOp

def symCNOT(U):
    '''CNOT circuit from binary invertible matrix U'''
    m,n = U.shape
    U2 = ZMatZeros((2*n,2*n))
    U2[:n,:n] = U
    Uinv = binMatInv(U.T)
    U2[n:,n:] = Uinv
    return U2

def symCX(n,i,j):
    '''CNOT_ij on n qubits as symplectic matrix'''
    C = ZMatI(n)
    C[j] ^= C[i]
    S = ZMatZeros((2*n,2*n))
    S[:n,:n] = C 
    S[n:,n:] = C.T
    return S

def symCZ(n,i,j):
    '''CZ_ij on n qubits as symplectic matrix
    CZ_ii = S_i'''
    Q = ZMatZeros((n,n))
    Q[i,j] = 1
    Q[j,i] = 1
    S = ZMatI(2*n)
    S[:n,n:] = Q
    return S

def symCXX(n,i,j):
    '''C(X,X)_ij on n qubits as symplectic matrix
    C(X,X)_ii = sqrt{X}_i'''
    return symCZ(n,i,j).T

def symKron(SList):
    '''Kronecker product of list of symplectic matrices'''
    nList = [len(S)//2 for S in SList]
    n = np.sum(nList)
    S = ZMatZeros((2*n,2*n))
    c = 0
    for ni,Si in zip(nList,SList):
        S[c:c+ni,c:c+ni] = Si[:ni,:ni]
        S[c:c+ni,n+c:n+c+ni] = Si[:ni,ni:]
        S[n+c:n+c+ni,c:c+ni] = Si[ni:,:ni]
        S[n+c:n+c+ni,n+c:n+c+ni] = Si[ni:,ni:]
        c += ni
    return S     

########################################################################
## Random Generation of GL2 (Binary Invertible Matrix) Operators
########################################################################


def GLRandVec(rng,n):
    '''generate a list of n random binary vectors which can be used to generate a GL matrix'''
    xList = []
    for i in range(n):
        done = False
        while not done:
            ## part of the vector must be non-zero - repeat until we have such a vector
            x = rng.integers(2,size=n)  
            done = np.sum(x[:n-i]) > 0
        xList.append(x)
    return xList

def vec2GL(xList):
    '''convert a list of binary vectors to a GL matrix'''
    r = len(xList)
    A = ZMatI(r)
    for i in range(r):
        x = xList[i]
        ## part of the vector must be non-zero
        j = min(bin2Set(x[:r-i]))
        if j > i:
            ## SWAP to ensure x[i] = 1
            x[j],x[i] = x[i],x[j]
            A[j] ^= A[i]
            A[i] ^= A[j]
            A[j] ^= A[i]
        for j in bin2Set(x):
            ## Add rows to ensure row is all zero, apart from component i
            if j != i:
                A[j] ^= A[i]
    return A

def vec2GL(xList):
    '''Convert a binary string to an element of GL_2 on r bits'''
    A = []
    r = len(xList)
    for i in range(r):
        x = xList[i]
        if len(A) == 0:
            A.append(x)
        else:
            H,p = getH(ZMat(A),2,retPivots=True)
            b = ZMatZeros(r)
            p1 = invRange(r,p)
            b[p1] = x[:r-i]
            uA = matMul(x[r-i:],A,2)[0]
            A.append(b ^ uA)
    return ZMat(A)

########################################################################
## Generation of Clifford Operators
########################################################################

def randomCAbin(rng,n,k):
    '''Generate random A and C matrices
    These represent linear combinations of stabilisers to add to a tableau
    to form  a logical Clifford operator with desired action'''
    r = n-k
    xList = GLRandVec(rng,r)
    xList = [rng.integers(2,size= (2 * r * k + r*(r+1)//2)), np.reshape(xList,-1) ]
    return np.hstack(xList)

def bin2CA(x,n,k):
    '''convert binary representation to C and A matrix'''
    r = n-k
    b =  r*(r+1)//2
    A1 = makeSymmetricMatrix(r,x[:b],Sdiag=True)
    a = b
    b = a + r*k
    A2 = np.reshape(x[a:b], (k,r))
    a = b
    b = a + r*k
    C2 = np.reshape(x[a:b], (k,r))
    a = b
    b = a+r*r
    xList = np.reshape(x[a:b],(r,r))
    C1 = vec2GL(xList)
    # print(func_name(),getH(C1,2))
    return ZMatVstack([C1,C2]),ZMatVstack([A1,A2])

def sym2UCA(T,k):
    '''convert a symplectic matrix to U,C,A matrix forms
    k is the number of logical qubits'''
    n = len(T) // 2
    r = n-k
    ## logical action
    U = ZMatZeros((2*k,2*k))
    U[:k,:k] = T[r:n,r:n]
    U[:k,k:] = T[r:n,n+r:]
    U[k:,:k] = T[n+r:,r:n]
    U[k:,k:] = T[n+r:,n+r:]

    ## C1: Invertible matrix - stab gen change of basis
    C1 = T[:r,:r]
    ## C2: stabilisers added to LX
    C2 = matMul(T[n:n+r,n+r:].T,C1,2)

    ## A2: stabilisers added to LZ
    A2 = matMul(T[n:n+r,r:n].T,C1,2)
    ## A1: symmetric matrix - stabs added to destabs
    A1 = matMul(C1.T,T[n:n+r,:r],2) ^ matMul(C2.T,A2,2)
    C,A = ZMatVstack([C1,C2]), ZMatVstack([A1,A2])

    # print('Sym2UCA check',np.sum(UCA2sym(U,C,A) ^ T)==0)
    return U, C, A

def UCA2sym(U,C,A):
    '''Convert U,C,A matrices to symplectic matrix'''
    n = len(C)
    k = len(U)//2
    r = n-k
    ## Calculate symplectic matrix IxU
    IxU = ZMatI(2*n)
    IxU[r:n,r:n] = U[:k,:k]
    IxU[r:n,n+r:] = U[:k,k:]
    IxU[n+r:,r:n] = U[k:,:k]
    IxU[n+r:,n+r:] = U[k:,k:]

    ## Calculate symplectic matrix UA
    UA = ZMatI(2*n)
    UA[n:,:r] = A
    A1 = A[:r]
    # print('A1 symmetric',0==np.sum(A1 ^ A1.T))
    A2 = A[r:].T
    UA[n:n+r,r:n] = A2

    ## Calculate symplectic matrix UC
    UC = ZMatI(2*n)
    UC[:n,:r] = C
    In,Cinv = getHU(UC[:n,:n],2)
    # print('C invertible',0 == np.sum(ZMatI(n) ^ In))
    UC[n:,n:] = Cinv.T

    ## output for debugging
    # print('IxU',isSymplectic(IxU))
    # print(ZMatPrint(IxU,tB=2))
    # print('UC',isSymplectic(UC))
    # print(ZMatPrint(UC,tB=2))
    # print('UA',isSymplectic(UA))
    # print(ZMatPrint(UA,tB=2))

    return matMul(IxU,matMul(UC,UA,2),2)


######################################################################
## Decomposition of Symplectic Matrix into T = UA @ UB @ UC @ UH
## UA: CXX and sqrt{X} operators
## UB: CZ and S operators
## UC: CNOT operators
## UH: Hadamard operators
######################################################################

def sym2ABCH(T):
    '''Decomposition of Symplectic Matrix into T = UA @ UB @ UC @ UH'''
    n = len(T)//2
    CB = T[:n]
    CB,pX = getH(CB,2,nC=n,retPivots=True)
    B2 = CB[len(pX):,n:]
    B2,h = getH(B2,2,retPivots=True)
    h = ZMat(h)
    TH = XZhad(T,h) if len(h) > 0 else T
    C = TH[:n,:n]
    In, Cinv = getHU(C,2) 
    B = matMul(TH[:n,n:],C.T,2)
    A = matMul(TH[n:,:n],Cinv,2)
    print('T test:', 0==np.sum(T ^ ABCH2sym(A,B,C,h)))
    return A,B,C,h

def ABCH2sym(A,B,C,h):
    '''convert A,B,C,h into a symplectic matrix'''
    n = len(A)
    UA = ZMatI(2*n)
    UA[n:,:n] = A
    UB = ZMatI(2*n)
    UB[:n:,n:] = B
    In,Cinv = getHU(C,2) 
    UC = ZMatZeros((2*n,2*n))
    UC[:n,:n] = C 
    UC[n:,n:] = Cinv.T
    TTest = matMul(matMul(UA,UB,2),UC,2)
    TTest = XZhad(TTest,h)
    return TTest

#####################################################
## Convert stabiliser codes into tableau format
#####################################################

def Stab2Tableau(S0):
    '''Return n,k tableau plus phases for stabilisers in binary form S0'''
    n = len(S0.T) // 2
    # print('S0')
    # print(ZMatPrint(S0,tB=2))
    ## RREF mod 2 - only consider first n columns, return pivots
    H, Li = getH(S0,2,nC=n,retPivots=True)
    S1 = lowWeightGens(S0,tB=2,pList=[0,0.1,0.11,0.12])
    # print('S1')
    # print(ZMatPrint(S1,tB=2))
    ## independent X checks
    r = len(Li)
    ## reorder rows so pivots are to LHS
    ix = ZMat(Li + invRange(n,Li))
    H = ZMatPermuteCols(H,ix,tB=2)
    ## Swap cols r to n from X to Z component
    H = XZhad(H,np.arange(r,n))
    ## RREF again
    H,Li = getH(H,2,nC=n,retPivots=True)
    # print(len(H),len(S1))
    ## number of independent Z checks
    s = len(Li) - r
    ## number of encoded qubits
    k = n - r - s
    ## reorder columns
    ix2 = Li + invRange(n,Li)
    ix = ix[ix2]
    H = ZMatPermuteCols(H,ix2,tB=2)
    ## swap back cols r to n from X to Z component
    H = XZhad(H,np.arange(r,n))
    ## Extract key matrices
    A2 = H[:r,r+s:n]
    C = H[:r,-k:]
    E = H[r:,-k:]
    ## Form LX/LZ
    LX = ZMatHstack([ZMatZeros((k,r)),E.T,ZMatI(k),C.T, ZMatZeros((k,s+k))])
    LZ = ZMatHstack([ZMatZeros((k,n)),A2.T,ZMatZeros((k,s)),ZMatI(k)])
    ## Form destabilisers
    Rx = ZMatHstack([ZMatZeros((r,n)),ZMatI(r),ZMatZeros((r,n-r))])
    Rz = ZMatHstack([ZMatZeros((s,r)),ZMatI(s),ZMatZeros((s,n+k))])
    R = ZMatVstack([Rx,Rz])
    ## return qubits to original order
    ixR = ixRev(ix)
    # H = ZMatPermuteCols(S1,ix,tB=2)
    T = ZMatVstack([H,LX,Rx,Rz,LZ])
    T = ZMatPermuteCols(T,ixR,tB=2)
    ## adjust tableau phases to match phases of original stabilisers
    pT = PauliDefaultPhases(T)
    pS0 = PauliDefaultPhases(S0)
    r0, U = HowResU(S0,H,2)
    for i in range(len(H)):
        p, xz = PauliProd(S0,pS0,U[i])
        pT[i] = p
    return n,k,pT,T


def CSS2Tableau(HX,HZ):
    mX,n = HX.shape
    mZ = len(HZ)
    S0 = ZMatZeros((mX + mZ,n*2))
    S0[:mX,:n] = HX 
    S0[mX:,n:] = HZ
    T = CodeTableau(S0)
    return T


def indepPaulis(S):
    '''Return an independent set of Paulis from S in symplectic form'''
    S = np.flip(S,axis=0)
    N=2
    ## RREF plus transformation matrix
    H, U = getHU(S,N,tB=2)
    ## K is a list of linear combinations of rows which result in zero
    ix = np.sum(H,axis=-1) == 0
    K = U[ix,:]
    ## RREF - so combinations of lowest weight rows are to the RHS
    K, LI = getH(K,N,retPivots=True)
    ix = invRange(len(S),LI)
    ix = np.flip(ix)
    S = S[ix,:]
    return S

def CodeTableau(S0):
    '''Return n,k tableau plus phases for stabilisers in binary form S0'''
    n = len(S0.T) // 2
    S0 = indepPaulis(S0)
    ## RREF mod 2 - only consider first n columns, return pivots
    H, Li = getH(S0,2,nC=n,retPivots=True)
    ## independent X checks
    r = len(Li)
    ## reorder rows so pivots are to LHS
    ix = ZMat(Li + invRange(n,Li))
    H = ZMatPermuteCols(H,ix,tB=2)
    ## Swap cols r to n from X to Z component
    H = XZhad(H,np.arange(r,n))
    ## RREF again
    H,Li = getH(H,2,nC=n,retPivots=True)
    ## number of independent Z checks
    s = len(Li) - r
    ## number of encoded qubits
    k = n - r - s
    ## reorder columns
    ix2 = Li + invRange(n,Li)
    ix = ix[ix2]
    H = ZMatPermuteCols(H,ix2,tB=2)
    ## swap back cols r to n from X to Z component
    H = XZhad(H,np.arange(r,n))
    ## Extract key matrices
    A2 = H[:r,r+s:n]
    C = H[:r,-k:]
    E = H[r:,-k:]
    ## Form LX/LZ
    LX = ZMatHstack([ZMatZeros((k,r)),E.T,ZMatI(k),C.T, ZMatZeros((k,s+k))])
    LZ = ZMatHstack([ZMatZeros((k,n)),A2.T,ZMatZeros((k,s)),ZMatI(k)])
    ## Form destabilisers
    Rx = ZMatHstack([ZMatZeros((r,n)),ZMatI(r),ZMatZeros((r,n-r))])
    Rz = ZMatHstack([ZMatZeros((s,r)),ZMatI(s),ZMatZeros((s,n+k))])
    R = ZMatVstack([Rx,Rz])
    ## Find transformation U from S0 to H
    S1 = ZMatPermuteCols(S0,ix,tB=2)
    r0, U = HowResU(S1,H,2)
    ## Apply transformation U.T to destabilisers
    R = matMul(U.T,R,2)
    ## return qubits to original order
    ixR = ixRev(ix)
    ## Stack to form tableau, return qubits to original order
    T = ZMatVstack([R,LX,S1,LZ])
    T = ZMatPermuteCols(T,ixR,tB=2)
    return T


def PauliProd(S,pList,u):
    '''Calculate sign and X/Z components of product of Pauli operators S and phases pList specified by binary vector u'''
    r,n2 = S.shape
    n = n2//2
    p,xz = 0, ZMatZeros(2*n)
    for i in bin2Set(u):
        p = p + pList[i] + 2 * (np.sum(xz[n:] * S[i][:n]) % 2)
        xz ^= S[i]
    return p % 4, xz

def PauliDefaultPhases(S):
    '''Calculate a phase correction which ensures Pauli operators have non-trivial +1 eigenspace'''
    r,n2 = S.shape
    n = n2//2
    return np.sum(S[:,:n] * S[:,n:], axis=-1) % 4

def PauliComm(a,A):
    '''Calculate commutator of Pauli operator a with list of Paulis A'''
    n = len(a)//2
    On = OmegaMat(n)
    p = matMul(ZMat2D(a),matMul(On,A.T,2),2)
    return p[0]