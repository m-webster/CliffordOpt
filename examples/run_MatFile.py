import concurrent.futures
from run_utils import *

if __name__ == '__main__':
    parser = defaultParser()
    params = parser.parse_args()
    params.infile = f'MatFiles/{params.file}.txt'
    set_global_params(params)
    write_params(params)

    UList = readMatFile(params.infile)
    myrange = range(len(UList))

    ## Test - run in serial
    # for i in myrange:
    #     synthSave(UList[i],i,params)
    
    ## Production - run in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        threadFuture = {executor.submit(synthSave,UList[i],i,params): (i+1) for i in myrange}
        for future in concurrent.futures.as_completed(threadFuture):
            i = threadFuture[future]
