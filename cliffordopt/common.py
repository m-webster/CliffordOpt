import numpy as np
import itertools as iter
import sys
import time
import collections


class paramObj(object):
    pass

def deep_getsizeof(o, ids=None):
    if ids is None:
        ids = set()
    d = deep_getsizeof
    if id(o) in ids:
        return 0
 
    r = sys.getsizeof(o)
    ids.add(id(o))
 
    if isinstance(o, str) or isinstance(0, str):
        return r
    
    if isinstance(o, dict):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())
 
    if isinstance(o, collections.abc.Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
 
    if isinstance(o, collections.abc.Container):
        return r + sum(d(x, ids) for x in o)
 
    return r 


#######################################
## Sorting and Indexing
#######################################

def argsort(seq,reverse=False):
    '''Argsort but allowing for sorting of tuples'''
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__,reverse=reverse)

def argmin(seq,reverse=False):
    '''Argsort but allowing for sorting of tuples'''
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    compfun = max if reverse else min
    return compfun(range(len(seq)), key=seq.__getitem__)

def ixRev(ix):
    ## return indices to restore original column order
    ## input: ix - permutation of [0..n-1]
    ## output: ixR such that ix[ixR] = [0..n-1]
    n = max(ix) + 1
    ixR = ZMatZeros(n)
    ixR[ix] = range(len(ix))
    return ixR



def nonDecreasing(w):
    '''Check whether vector w is non-decreasing'''
    for i in range(1,len(w)):
        if w[i] < w[i-1]:
            return False 
    return True

def printObj(params):
    temp = []
    for k,v in vars(params).items():
        temp.append(f'{k}: {v}')
    return "\n".join(temp)

#######################################
## ZMat - Integer Matrices
#######################################

#######################################
## Create ZMat
#######################################

def ZMat(A,n=None):
    '''Create an integer numpy array. If n is set, ensure that the row length is n.'''
    if typeName(A) in ['set','range']:
        A = list(A)
    if typeName(A) != 'ndarray' or A.dtype != int:
        A = np.array(A,dtype=int)
    if n is not None:
        s = list(A.shape)
        if s[-1] == 0:
            A= np.empty((0,n),dtype=int)
    return A

def ZMatI(n):
    '''Identity n x n integer matrix'''
    return np.eye(n,dtype=int)

def ZMatZeros(s):
    '''Return integer array of zeros of length/shape s'''
    return np.zeros(s,dtype=int)


def makeSymmetricMatrix(r,x,Sdiag=True):
    '''convert binary vector x to a symmetrix rxr matrix
    if Sdiag is False, the diagonal is all zeros'''
    S = ZMatZeros((r,r))
    if Sdiag:
        S[np.triu_indices(r, 0)] = x
    else:
        S[np.triu_indices(r, 1)] = x
    S[np.tril_indices(r, -1)] = S.T[np.tril_indices(r, -1)]
    return S


###################################################
## Calc Linear Combinations of up to t rows
###################################################

def Orbit2distIter(SX,t=None,return_u=False):
    '''Interator yielding binary rows of form (u SX mod 2) for wt(u) <= t.
    if return_u, yield u as well as the row.'''
    r, n = np.shape(SX)
    if t is None:
        t = r
    t = min(t, r)
    for k in range(t+1):
        for xSupp in iter.combinations(range(r),k):
            vSX = np.mod(np.sum(SX[xSupp,:],axis=0),2)
            if return_u:
                u = set2Bin(r,xSupp)
                yield vSX, u
            else:
                yield vSX

# def Orbit2dist(SX,t=None,return_u=False):
#     '''Matrix with binary rows of form (q + u SX mod 2) for wt(u) <= t.
#     if return_u, yield u as well as the row.'''
#     temp = list(Orbit2distIter(SX,t,return_u))
#     # temp = Orbit2dist(SX,t,return_u)
#     print(temp)
#     if return_u:
#         temp = list(zip(*temp))
#         return [ZMat(a) for a in temp]
#     else:
#         return ZMat(temp)

def Orbit2dist(SX,t=None,return_u=False):
    r,n = SX.shape
    if t is None:
        t = r
    t = min(t, r)
    if return_u:
        AList = binLinComb(ZMatHstack([SX,ZMatI(r)]),t)
        A = ZMatVstack(AList)
        return [A[:,:n],A[:,n:]]
    AList = binLinComb(SX,t)
    return ZMatVstack(AList)
    
binomCoeff = dict()

def binom(n,k):
    # print('binom',n,k)
    if k > n or k < 0 or n < 0:
        return 0
    global binomCoeff
    ix = (n,k)
    if ix not in binomCoeff:
        if k == 0 or n-k == 0:
            binomCoeff[ix] = 1
        elif k == 1 or n-k == 1:
            binomCoeff[ix] = n
        else:
            binomCoeff[ix] = binom(n-1,k-1) + binom(n-1,k)
    return binomCoeff[ix]

def binLinCombRec(c,d,AList,populated):
    '''Recursive step for binLinComb - update lin comb of d of the first c rows of A into AList'''
    ## Can't form lin comb if c < d; if c < 1, nothing to do
    if c < d or c < 1:
        return
    ## Check if we have already populated the relevant part of AList
    if populated[d] < c:
        ## chose d rows from first c-1 rows of A
        if c > d and populated[d] < c-1:
            binLinCombRec(c-1,d,AList,populated)
        ## choose d-1 rows from first c-1 rows of A, then add row A[c-1]
        if d > 1:
            if populated[d-1] < c-1:
                binLinCombRec(c-1,d-1,AList,populated)  
            ## update AList
            AList[d][binom(c-1,d): binom(c-1,d)+binom(c-1,d-1) ] = AList[d-1][:binom(c-1,d-1)] ^ AList[1][c-1]
        ## update populated
        populated[d] = c

def binLinComb(A,d):
    '''Linear combinations of up to d rows of A'''
    r,n = A.shape
    ## Allocate space
    AList = [np.empty((binom(r,k),n),dtype=np.int8) for k in range(d+1)]
    ## AList[0] is the all zero vector
    AList[0][:,:] = 0
    ## AList[1] is A
    AList[1] = A
    ## keep track of which parts of AList have been populated
    populated = [n] * 2 + [0] * (d-1)
    for k in range(2,d+1):
        binLinCombRec(r,k,AList,populated)
    return AList

def getdMax(r,maxLen=1 << 20):
    s = 0
    for i in range(r+1):
        s += binom(r,i)
        if s > maxLen:
            # print(func_name(),s,i-1)
            return i-1
    return r

def weightEnumerator(A):
    m,n = A.shape
    temp = ZMatZeros(n+1)
    for s in Orbit2dist(A):
        temp[np.sum(s)] += 1
    return temp

####################################
## Conversions to other formats
####################################

def isIter(A):
    return hasattr(A,'__iter__')

def Lolshape(CC):
    if len(CC) == 0 or not isIter(CC[0]):
        return [len(CC)]
    else:
        return [len(CC)] + Lolshape(CC[0])

def SL2ZM(CC,ix,A):
    '''recursive step for Sets2ZMat'''
    if len(CC) == 0 or not isIter(CC[0]):
        for i in CC:
            ix2 = (ix + [i])
            A[tuple(ix2)] = 1
    else:
        i = 0
        for c in CC:
            ix2 = ix + [i]
            SL2ZM(c,ix2,A)
            i+=1

def Sets2ZMat(n,CC):
    '''Convert a list of lists or sets to a binary matrix with rows of length n
    Fast method - matrix storage is allocated at beginning'''
    ## get shape of final matrix
    s = Lolshape(CC)
    s[-1] = n
    ## intialise to all zeros
    A = np.zeros(s,dtype=np.int8)
    ## recursion
    SL2ZM(CC,[],A)
    return A

def ZMat2Sets(A):
    '''Convert a binary matrix to a list of lists of non-zero entries
    recursive method'''
    if len(A.shape) == 1:
        return bin2Set(A)
    else:
        return [ZMat2Sets(B) for B in A]

def ZMatVstack(AList):
    '''Faster method for stacking 2xD matrices'''
    if len(AList) == 0:
        return ZMatZeros((0,0))
    AList = [ZMat(A) for A in AList]
    ALen = [len(A) for A in AList]
    m = np.sum(ALen)
    A = AList[0]
    r,n = A.shape
    B = np.empty((m,n),dtype=A.dtype)
    c1 = 0
    for i in range(len(AList)):
        c2 = c1 + ALen[i]
        B[c1:c2] = AList[i]
        c1 = c2
    return B

def ZMatHstack(AList):
    '''Faster method for stacking 2xD matrices'''
    # return ZMatVstack([A.T for A in AList]).T
    if len(AList) == 0:
        return ZMatZeros((0,0))
    AList = [ZMat(A) for A in AList]
    ALen = [len(A.T) for A in AList]
    m = np.sum(ALen)
    A = AList[0]
    r,n = A.shape
    B = np.empty((r,m),dtype=A.dtype)
    c1 = 0
    for i in range(len(AList)):
        c2 = c1 + ALen[i]
        B[:,c1:c2] = AList[i]
        c1 = c2
    return B

def Mnt(n,t,mink=1):
    '''Rows are binary strings of length n of weight mink to t'''
    CC = [s for k in range(mink, t+1) for s in iter.combinations(range(n),k)]
    return Sets2ZMat(n,CC)
    A = [set2Bin(n,s) for k in range(mink, t+1) for s in iter.combinations(range(n),k)]
    return ZMat(A)

### multi dimensional versions?
def set2Bin(n,A):
    '''Convert list of integers t to a binary vector of length n'''
    temp = ZMatZeros(n)
    temp[list(A)] = 1
    return temp

def bin2Set(v):
    '''Convert binary vector to a list of indices such that v[i] !=0'''
    v = np.ravel(v)
    return list(np.nonzero(v)[0])

def Zmat2tuple(A):
    '''Convert rows of A to tuples.'''
    n = np.shape(A)[-1]
    A = np.reshape(A,(-1,n))
    return [tuple(a) for a in A]

def set2tuple(c):
    return tuple(sorted(set(c)))
    n = max(c) + 1
    return tuple(inRange(n,c))

def invRange(n,S):
    '''return list of elements of range(n) NOT in S'''
    return sorted(set(range(n)) - set(S))

def inRange(n,S):
    '''return list of elements of range(n) which ARE in S'''
    return sorted(set(S).intersection(range(n)))

def int2bin(x,d,N=2):
    '''convert integer to base N vector'''
    temp = []
    for i in range(d):
        temp.append(x % N)
        x = x // N
    return ZMat(temp)

##############################
## ZMat Info
##############################

def leadingIndex(a):
    '''Return leading index of vector a (ie smallest value for which a[i] !=0)'''
    i = 0
    n = len(a)
    while i < n and a[i]==0:
        i+=1
    return i

def isZero(A,N=False):
    '''Check if A modulo N = 0 for all values in A.'''
    if N:
        A = np.mod(A,N)
    return np.all(A == 0)


#######################################
## ZMat Analysis
#######################################

def freqTable(wList):
    '''Dict of val:count for val in wList'''
    temp = {w:0 for w in set(wList)}
    for w in wList:
        temp[w] += 1
    return temp

def freqTablePrint(wList):
    FT = freqTable(wList)
    temp = [f'{k}:{FT[k]}' for k in sorted(FT.keys())]
    return ",".join(temp)

##########################
## Changing Shape of ZMat
##########################

def ZMat2D(A):
    '''Return a 2-dimensional integer numpy array from A.'''
    A = ZMat(A)
    if np.ndim(A) == 2:
        return A
    if np.ndim(A) == 0:
        return ZMat([[A]])
    if np.ndim(A) == 1:
        return ZMat([A])    
    d = np.shape(A)[-1]
    return np.reshape(A,(-1,d))

##########################
## String I/0 for ZMat
##########################

def row2components(r):
    '''For integer vector r return indices for the non-zero values ix=supp(r) and the non-zero values r[ix].
    Useful for displaying large vectors.'''
    ix = ZMat(np.nonzero(r))
    return ix, r[ix]

def row2compStr(r):
    '''Display row r using indices for non-zero values and list of non-zero values
    Useful for displaying large vectors.'''
    ix,vals = row2components(r)
    return f'{ix}:= {vals}'.replace(" ","")

def ZMat2compStr(A):
    '''Display 2D integer matrix A using indices for non-zero values and list of non-zero values
    Useful for displaying large vectors.'''
    return "\n".join([row2compStr(r) for r in A])

def str2ZMat(mystr):
    '''Convert string of single digit numbers or multi digit numbers split by spaces to an integer array'''
    if mystr.find(" ") > 0:
        mystr = mystr.split()
    return ZMat([int(s) for s in mystr])

def str2ZMatdelim(S=''):
    '''Convert string with rows separated by \r, \n "," or ; to 2D integer array.'''
    sep=','
    for s in "\r\n;":
        S = S.replace(s,sep)
    S = S.split(sep)
    return ZMat([str2ZMat(s) for s in S])

def bin2ZMat(SX):
    '''Convert multiple types of binary vector input to integer matrix.
    SX is either string or array.'''
    if SX is None:
        return SX 
    ## convert string to ZMat
    if isinstance(SX,str):
        return str2ZMatdelim(SX.strip())
    ## convert array to ZMat
    return ZMat(SX)

def ZMat2str(A,N=None):
    '''Return string version of integer matrix A.'''
    if np.size(A) == 0:
        return ""
    S = np.char.mod('%d', A)
    sep = ""
    if N is None:
        N = np.amax(A) + 1
    if N > 10:
        Nw= len(str(N-1))
        S = np.char.rjust(S,Nw)
        sep = " "
    return sep.join(S)
    return np.apply_along_axis(func1d=sepjoin,axis=-1,arr=S,sep=sep)

def sepjoin(a,sep):
    '''Join text vector a using sep - for display of ZMat.'''
    return sep.join(a)

def ZMatPrint(A,N=None,nA=0,tB=1):
    '''Print integer matrix A'''
    temp = []
    A = ZMat2D(A)
    m,n = A.shape
    nB = (n - nA)//tB
    for r in A:
        myRow = [ZMat2str(r[t*nB:(t+1)*nB],N) for t in range(tB)]
        if tB*nB < n:
            myRow.append(ZMat2str(r[tB*nB :],N))
        temp.append("|".join(myRow))
    return "\n".join(temp)

#################################################
## Debugging Functions
#################################################

def currTime():
    '''Return current time'''
    return time.time()

def startTimer():
    '''Start timer for algorithm and set global variable startTime to be the current time.'''
    global startTime
    startTime = time.time()
    return startTime

def elapsedTime():
    '''Return the time elapsed from last startTimer() call.'''
    global startTime
    return -startTime + startTimer()

def func_name():
    """Return the name of the current function - for debugging."""
    return sys._getframe(1).f_code.co_name

def typeName(val):
    '''Return the name of the type of val in text form.'''
    return type(val).__name__


##################################
## Integer logarithms
##################################

def logCeil(x,N=2):
    '''Return min(t) where x <= N^t'''
    i = 0
    while x > 0:
        x = x // N
        i = i+1
    return i

def log2int(N):
    '''Find t such that N = 2**t or None otherwise'''
    t = logCeil(N-1,2) 
    return t if 2 ** t == N else None
