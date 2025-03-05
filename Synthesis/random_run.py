import add_parent_dir
import numpy as np
from CliffordOps import *
from clifford_synthesis import *
from NHow import *
import concurrent.futures


if __name__ == '__main__':
    parser = defaultParser()
    params = parser.parse_args()
    params.infile = f'Synthesis/random/{params.file}.txt'
    set_global_params(params)
    write_params(params)

    UList = readMatFile(params.infile)

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        myrange = range(len(UList))
        myrange = range(1)
        threadFuture = {executor.submit(random_run,UList,i,params): (i+1) for i in myrange}
        for future in concurrent.futures.as_completed(threadFuture):
            i = threadFuture[future]
