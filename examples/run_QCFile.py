import concurrent.futures
from run_utils import *

if __name__ == '__main__':
    parser = defaultParser()
    params = parser.parse_args()
    params.mode = 'QC'
    params.infile = f'QCFiles/{params.file}'
    set_global_params(params)
    write_params(params)

    circuitList, circuitNames = readQCFile(params.infile)
    myrange = range(len(circuitList))
    # myrange = range(32,len(circuitList))
    ixMin = params.ixMin
    ixMax = params.ixMax if params.ixMax > 0 else len(circuitList)
    ixMax = min(ixMax, len(circuitList))
    ## Test - run in serial
    for i in range(ixMin,ixMax):
        synthSave(circuitList[i],i,params,circuitNames[i])

    ## Production - run in parallel
    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    #     threadFuture = {executor.submit(synthSave,circuitList[i],i,params,circuitNames[i]): (i+1) for i in myrange}
    #     for future in concurrent.futures.as_completed(threadFuture):
    #         i = threadFuture[future]
