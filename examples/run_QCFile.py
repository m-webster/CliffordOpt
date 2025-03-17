import concurrent.futures
from run_utils import *

if __name__ == '__main__':
    parser = defaultParser()
    params = parser.parse_args()
    params.mode = 'QC'
    params.infile = f'QCFiles/{params.file}.csv'
    set_global_params(params)
    write_params(params)

    circuitList, circuitNames = readQCFile(params.infile)
    myrange = range(len(circuitList))

    ## Test - run in serial
    # for i in myrange:
    #     synthSave(circuitList[i],i,params,circuitNames[i])

    ## Production - run in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        threadFuture = {executor.submit(synthSave,circuitList[i],i,params,circuitNames[i]): (i+1) for i in myrange}
        for future in concurrent.futures.as_completed(threadFuture):
            i = threadFuture[future]
