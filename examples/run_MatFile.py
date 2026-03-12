import concurrent.futures
from run_utils import *

if __name__ == '__main__':
    parser = defaultParser()
    params = parser.parse_args()
    if params.file.upper()[:2] == "GL":
        params.mode = "GL"
    params.infile = f'MatFiles/{params.file}'
    set_global_params(params)
    write_params(params)

    UList = readMatFile(params.infile)
    ixMin = params.ixMin
    ixMax = params.ixMax if params.ixMax > 0 else len(UList)
    ixMax = min(ixMax, len(UList))
    ## Test - run in serial
    for i in range(ixMin,ixMax):
        synthSave(UList[i],i,params)
    
    ## Production - run in parallel
    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    #     threadFuture = {executor.submit(synthSave,UList[i],i,params): (i+1) for i in myrange}
    #     for future in concurrent.futures.as_completed(threadFuture):
    #         i = threadFuture[future]

