from cliffordopt import *

#####################################################################
## Functions for Databases of Optimal Circuits
#####################################################################

if __name__ == '__main__':
    params = paramObj()
    n = 4                       ## number of qubits
    params.mode = 'GL'          ## either GL for CNOT circuits or Sp for general Clifford circuits
    params.minDepth = True     ## optimal depth = True or optimal gate-count = False
    nWorkers = 8                ## number of threads to run simultaneously

    #####################################################################
    ## Generate  Database for Optimal Clifford Synthesis
    #####################################################################
    ## Production mode
    ResumeDBGen(n,params.mode,nWorkers,params.minDepth )
    ## Test mode
    # cProfile.run('ResumeDBGen(n,params.mode,nWorkers,params.minDepth)')

    #####################################################################
    ## opt DB analysis by depth/gate count 
    #####################################################################
    print("\n")
    print(f'{params.mode}({n},2) {"min depth" if params.minDepth else "min op-count"}')
    print('d\tcount')
    for d, count in analyseOptDB(n,params.mode,params.minDepth):
        print(f'{d}\t{count}')

    #####################################################################
    ## opt DB correlation
    #####################################################################
    # lCorr,sCorr = correlDB(params.mode,n,params.minDepth)
    # print(f'Log Corr: a,b,R: {lCorr[0]:.2f}\t{lCorr[1]:.2f}\t{lCorr[2]:.2f}')
    # print(f'Sum Corr: a,b,R: {sCorr[0]:.2f}\t{sCorr[1]:.2f}\t{sCorr[2]:.2f}')
