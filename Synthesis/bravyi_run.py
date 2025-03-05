import add_parent_dir
import numpy as np
from CliffordOps import *
from clifford_synthesis import *
from NHow import *
import concurrent.futures

if __name__ == '__main__':
    parser = defaultParser()
    params = parser.parse_args()
    params.mode = 'sym'
    params.infile = f'Synthesis/bravyi/{params.file}.csv'
    set_global_params(params)
    write_params(params)

    circuits = readBravyiFile(params.infile)

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        myrange = range(len(circuits))
        threadFuture = {executor.submit(bravyi_run,circuits,i,params): (i+1) for i in myrange}
        for future in concurrent.futures.as_completed(threadFuture):
            i = threadFuture[future]
