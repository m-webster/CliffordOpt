
from cliffordopt import *
import numpy as np

#####################################################################
## CNOT synthesis methods used in Low-overhead Magic State Circuits with Transversal CNOTs https://arxiv.org/abs/2501.10291
#####################################################################

def SWAP2end(layers,n):
    temp = []
    ixC = np.arange(n)
    ixR = ixC
    for opType,opData in layers:
        if opType == 'SWAP':
            ixC = ixC[opData]
            ixR = ixRev(ixC)
        if opType == 'T':
            temp.append(('T',opData[ixR]))
        if opType == 'CX':
            temp.append(('CX',[(ixC[a],ixC[b]) for (a,b) in opData]))
    if np.any(ixC != np.arange(n)):
        temp.append(('SWAP',ixC))
    return temp

def SWAP2front(layers,n):
    ixC = np.arange(n)
    for opType,opData in layers:
        if opType == 'SWAP':
            ixC = ixC[opData]
    if np.any(ixC != np.arange(n)):
        ixR = ixRev(ixC)
        temp = SWAP2end([('SWAP',ixR)] + layers,n)
        temp = [('SWAP',ixC)] + temp 
    else:
        temp = SWAP2end(layers,n)
    return temp

def partRTCheck(UList,qList,layers):
    # print('Phase Rotation Matrix+phases')
    # print(phaseRotPrint(UList,qList))
    EList = lengthnvectors(n)
    pList = ZMatZeros(len(EList))
    for opType, op in layers:
        # print(opType,op)
        if opType == 'CX':
            for j,k in op:
                EList[:,k] ^= EList[:,j]
        if opType == 'T':
            pList += matMul(op,EList.T,8)[0]
        if opType == 'SWAP':
            EList = EList[:,op]
    pList = np.mod(pList,8)
    # print('Circuit Action')
    # print(phaseRotPrint(EList,pList))
    pList2 = phaseRotTest(UList,qList)
    return np.all(pList == pList2)

def ParRTList(UList,qList,optCX=CNOT_greedy_depth):
    m,n = UList.shape
    U = UList[:n]
    q = qList[:n]
    layers = []
    gateCount = 0
    if not indepRows(U):
        print('Rows of U not independent')
        return False

    ix, opList = optCX(U.T)
    layers.append(('SWAP',ix))
    layers.append(('CX',opList))
    
    a = 0
    V = None
    for i in range(m//n):
        b = min(a+n,m)
        U = UList[a:b]
        if not indepRows(U):
            print('Rows of U not independent')
            return False
        q = qList[a:b]
        if V is not None:
            UV = matMul(U,V,2)
            ix, opList = optCX(UV.T)
            layers.append(('SWAP',ix))
            layers.append(('CX',opList))
            gateCount += len(opList)
        V = binMatInv(U)
        layers.append(('T',q))
        a = b
    ix, opList = optCX(V.T)
    layers.append(('SWAP',ix))
    layers.append(('CX',opList))
    gateCount += len(opList)
    for a,b in layers:
        print(a,b)
    layers = SWAP2front(layers,n)
    print('#####################')
    print(f'CX Count: {gateCount}')
    for a,b in layers:
        print(a,b)
    print('Check:',partRTCheck(UList,qList,layers))
    return layers,gateCount

def indepRows(A):
    m1 = len(A)
    A = RemoveZeroRows(A)
    m,n = A.shape
    if (m1 > m) or (m > n):
        return False
    if m <= 1:
        ## 0 or 1 row - always independent if non-zero
        return True
    H = getH(A,2)
    if m1 > len(H):
        return False
    return True

def lengthnvectors(n):
    In = ZMatI(n)
    return Orbit2dist(In,n)

def phaseRotTest(U,qList,verbose=False):
    m,n = U.shape
    EList = lengthnvectors(n)
    pList = matMul(EList,U.T,2)
    pList = matMul(qList,pList.T,8)[0]
    if verbose:
        print('Phase Rotations')
        print(ZMatPrint(ZMatHstack([EList,ZMat2D(pList).T]),nA=1))
    return pList

def phaseRotPrint(UList,qList):
    return ZMatPrint(ZMatHstack([UList,ZMat([qList]).T]),nA=1)


## Fig 5: CCZ Distillation
U = '''1001
0110
0010
1110
1010
0101
0001
1101'''
qList = '17117177'
ix = [4,5,7,0,3,1,2,6]

## Fig 5: CCZ update 20241223
U = '''1011
0111
0011
1111
1001
0101
0001
1101'''
qList = '77111177'
ix = [0, 1, 3, 4, 2, 6, 5, 7]


# ## Fig 8: T Distillation
# U = '''01000
# 11100
# 10110
# 01011
# 01110
# 00010
# 00111
# 01101
# 11010
# 11001
# 10000
# 00100
# 10011
# 10101
# 11111'''
# qList = '111111111111111'
# # ix = [ 9,  6, 14,  4,  1,  7, 13,  8,  3,  2, 12, 10,  5, 11,  0] # count 16
# ix = [ 7,  2,  3, 13, 14,  9,  4,  1,  8, 12, 11,  5, 10,  0,  6] # count 17, depth 11


# CS Distillation Fig 12
U = '''1111
0110
1010
1110
0001
0101
1001
1011
0011
0111
1101
0010'''
qList = '111771171777'
ix = [2,3,0,8,9,5,10,7,6,11,1,4]

## 4T gates -> 1 |CCZ> state
# U = '''001
# 011
# 101
# 111
# 001
# 100'''
# qList = '177100'

U = bin2ZMat(U)
qList = str2ZMat(qList)

m,n = U.shape

print('Permutation',ix)
print('Phase Rotations')
print(phaseRotPrint(U[ix],qList[ix]))
ParRTList(U[ix],qList[ix])

# for i in range(2):
#     ix = np.random.permutation(range(m))
#     # ix = np.arange(m)
# for ix in iter.permutations(range(m)):
#     ix = ZMat(ix)
#     print(f'\nPermutation: {ix}')
#     ParRTList(U[ix],qList[ix])

CNOT_greedy(U[ix[:n]].T,verbose=True)
# print('Phase Rotation U')
# print(ZMatPrint(U[ix[:n]]))
# print('T', qList[ix[:n]])