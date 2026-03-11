import concurrent.futures
from run_utils import *
import os


def filterNonClifford(mytext):
    '''filter out non-Clifford RZ operators'''
    qcQASM = []
    pi = 3.14159
    for myRow in mytext.split("\n"):
        valid = True
        myText = myRow.strip().replace(" ","").lower()
        if len(myText) > 5 and myText[:3] in {"rx(","ry(","rz("}:
            b = myRow.find(')',3)
            theta = eval(myText[3:b])
            theta = 2*theta/pi
            thetaRound = round(theta)
            if (abs(theta-thetaRound) > 1e-3):
                valid = False
            # print(myRow,theta,valid)
        if len(myText) > 1 and myText[0] == 't':
            valid = False
        if valid:
            qcQASM.append(myRow)
    return "\n".join(qcQASM)

def readQCFolder(myFolder):
    '''read circuits from CSV file in Bravyi format - each circuit is a QASM string'''
    circuitList,circuitNames = [],[]
    for myFile in sorted(os.listdir(myFolder)):
        if myFile[-4:].lower() == 'qasm':
            with open(f'{myFolder}/{myFile}','r') as f:
                mytext = filterNonClifford(f.read())
                circuitList.append(mytext)
                circuitNames.append(myFile)
    return circuitList, circuitNames

if __name__ == '__main__':
    parser = defaultParser()
    params = parser.parse_args()
    params.mode = 'QC'
    params.method = 'greedy'
    params.file = f'SATCircuits/Random-Clifford/qsynth_0cost_swaps'
    set_global_params(params)
    write_params(params)

    circuitList, circuitNames = readQCFolder(params.file)
    # print(circuitNames)
    myrange = range(len(circuitList))
    # myrange = range(32,len(circuitList))

    ## Test - run in serial
    for i in myrange:
        synthSave(circuitList[i],i,params,circuitNames[i])

    ## Production - run in parallel
    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    #     threadFuture = {executor.submit(synthSave,circuitList[i],i,params,circuitNames[i]): (i+1) for i in myrange}
    #     for future in concurrent.futures.as_completed(threadFuture):
    #         i = threadFuture[future]
