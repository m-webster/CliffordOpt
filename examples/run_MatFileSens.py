import copy
import concurrent.futures
from run_utils import *

if __name__ == '__main__':
    parser = defaultParser()
    parser.add_argument("-r0", help="For astar range only: r0 - lowest r",type=float, default=2.2)
    parser.add_argument("-r1", help="For astar range only: r1 - highest r",type=float, default=1.9)
    parser.add_argument("--rStep", help="For astar range only: step size",type=float, default=1.9)
    params = parser.parse_args()
    params.astarRange = 1
    params.method = 'astar'
    params.hv = 0
    params.infile = f'MatFiles/{params.file}.txt'
    temp = params.file.split("_")
    params.t = temp[0]
    set_global_params(params)
    write_params(params)

    UList = readMatFile(params.infile,params.t)
    UList = [rec[0] for rec in UList]

    r0s = int(np.round(params.r0/params.rStep,1))
    r1s = int(np.round(params.r1/params.rStep,1))
    if r0s > r1s:
        r0s,r1s = r1s,r0s
    rVals = [i*params.rStep for i in range(r0s,r1s+1)]
    paramList = []
    for r in rVals:
        p1 = copy.copy(params)
        p1.hr = r
        paramList.append(p1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        myrange = range(len(UList))
        threadFuture = {executor.submit(synthSave,UList[i],i,p2): (i+1) for p2 in paramList for i in myrange }
        for future in concurrent.futures.as_completed(threadFuture):
            i = threadFuture[future]
