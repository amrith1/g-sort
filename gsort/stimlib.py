#This module is for analyzing results of electrical stimulation experiments
import os
from scipy.io import loadmat
import numpy as np
import glob
#import stimvis as sv
#from eilib import *
#import eilib as eil
from scipy.optimize import curve_fit
import tempfile
import subprocess
import re

def sigmoid(xs,m,b):
    return 1/ (1 + np.exp(-m*(xs - b)))


def getNotebook(dataset):
    #Definitions
    homedir = '/Volumes/Data/Notebooks/'

    #String games
    dataset_vec = dataset.split('-')
    fooDS = '-'.join(dataset_vec[:-1])
    pieceNum = dataset_vec[-1]

    #Convert to doc file
    ##Create temp name
    temp_name = './'+next(tempfile._get_candidate_names())
    temp_rtf_name = temp_name+'.rtf'
    temp_txt_name = temp_name+'.txt'
    FNULL = open(os.devnull, 'w') #supress output
    subprocess.call(['cp',homedir+fooDS+'.rtfd/TXT.rtf',temp_rtf_name])
    subprocess.call(['soffice', '--headless','--convert-to', 'txt:Text',temp_rtf_name],stdout=FNULL, stderr=subprocess.STDOUT)
    ##Read in text file
    with open(temp_txt_name, 'r') as f:
        showFlag = False
        for row in f:
            if fooDS+'-'+pieceNum in row:
                showFlag = True
            elif fooDS+'-'+str(int(pieceNum)+1) in row:
                showFlag = False
            elif fooDS+'-'+str(int(pieceNum)+2) in row:
                showFlag = False
            elif fooDS+'-'+str(int(pieceNum)+3) in row:
                showFlag = False #three numbers insurance
            if showFlag:
                cleanedRow = re.sub('[\n\r]', '', row)
                if cleanedRow:
                    print(cleanedRow)
    #Delete
    os.remove(temp_rtf_name)
    os.remove(temp_txt_name)


def getCellThresh(cellIds, matPath): #WRONGLY NAMED
	elecRespList = []
	elecRespNameList = glob.glob(matPath+'*mat')
	for matFile in elecRespNameList: 
		elecRespList.append(loadmat(matFile)['elecRespAuto'][0,0])
	return elecRespList


def getActVals(pattern, neuronID, elecRespSummaryPath):
	#Pattern is actually electrode - 1
	ersumm = loadmat(elecRespSummaryPath)['elecRespSummary']
	idx = list(ersumm[pattern,0]).index(neuronID)
	probs = ersumm[pattern,3][0][idx][0]
	amps = ersumm[pattern,3][0][idx][1]
	return amps,probs

def getActThresh(amps, probs):
	popt, pcov = curve_fit(sv.sigmoid, amps, probs)
	thr = sv.invsigmoid(0.5,*popt)
	return thr

def getActCurve(amps, probs, xvals):
    popt, pcov = curve_fit(sv.sigmoid, amps, probs)
    y = sv.sigmoid(xvals,*popt)
    return y

def getERThresh(ERPath, finalizedFlg=True, cutTrailingZeros = True):
    #Gets neuron threshold as manually set in elecResp file
    data = loadmat(ERPath)['elecResp']
    #Get amplitudes
    try:
        amps = np.abs(data['stimInfo'][0,0]['stimAmps'][0,0].flatten())
    except IndexError:
        print('elecResp file is empty!')
        return 0 
    probs = data['analysis'][0,0]['successRates'][0,0].flatten()
    finalized = data['analysis'][0,0]['finalized'][0,0].flatten()
    if finalizedFlg:
        if len(np.nonzero(finalized)[0]):
            if len(np.nonzero(finalized)[0])%2 == 1:
                finalized[np.nonzero(finalized)[0][-1]] = 0
            amps = amps[finalized > 0]
            probs = probs[finalized > 0]
            #Remove duplicate amplitudes and average values
            #if amps[0] == amps[1]:
            #    amps = np.array([np.abs(amps[i]) for i in range(0,len(amps),2)]).flatten()
            #    probs = np.array([[(probs[i] + probs[i+1])/2] for i in range(0,len(probs),2)]).flatten()
    amps, probs = removeAmpsProbsDupl(amps, probs)
    thr = getActThresh(amps, probs)
    if thr:
        print('found threshold')
    return thr



def getERAmpsProbs(ERPath, finalizedFlg=True, cutTrailingZeros = True):
#Gets neuron threshold as manually set in elecResp file
    data = loadmat(ERPath)['elecResp']
#Get amplitudes
    try:
        amps = data['stimInfo'][0,0]['stimAmps'][0,0].flatten()
    except IndexError:
        return (0,0)
    probs = data['analysis'][0,0]['successRates'][0,0].flatten()
    finalized = data['analysis'][0,0]['finalized'][0,0].flatten()
    if finalizedFlg:
        if len(np.nonzero(finalized)[0]):
            if len(np.nonzero(finalized)[0])%2 == 1:
                finalized[np.nonzero(finalized)[0][-1]] = 0
            amps = amps[finalized > 0]
            probs = probs[finalized > 0]
            #Remove duplicate amplitudes and average values
            #if amps[0] == amps[1]:
            #    amps = np.array([np.abs(amps[i]) for i in range(0,len(amps),2)]).flatten()
            #    probs = np.array([[(probs[i] + probs[i+1])/2] for i in range(0,len(probs),2)]).flatten()
            amps, probs = removeAmpsProbsDupl(amps, probs)

        return (amps, probs)

def getNumFinalizedER(ERDirPath):
    #Get list of elecRespSummary files
    erlist = glob.glob(ERDirPath + '/e*n*p*mat')
    total_len = len(erlist)
    final_num = 0

    #Load each elecResp file
    for er in erlist:
        data = loadmat(er)['elecResp']

        #Check for empty
        try:
            finalized = data['analysis'][0,0]['finalized'][0,0].flatten()
        except IndexError:
            continue

        #If number finalized is less than half, consider empty
        total_finalized_len = len(finalized) #same as number of amps ofc
        if len(np.nonzero(finalized)[0]) > total_finalized_len/2:
            final_num += 1

    return (final_num,total_finazlied_len)


def getMinElec(eis, nid):
    ei = np.amin(eis[nid,0],axis=1)
    return np.argmin(ei)
def classifyElecSomaAxon(data_basename, neuron_ids, elecsarr):
    thr = -1
    dist_thr = 200


    dr = loadmat(data_basename + '_forpy.mat')
    eis = dr['datarun'][0,0]['ei']['eis'][0,0]
    cellIds = list(dr['datarun'][0,0]['cell_ids'][0,:])
    
    #Get neuron index
    output = []
    for nn, neuron_id in enumerate(neuron_ids):
        nid = cellIds.index(neuron_id)
        elecs = elecsarr[nn]
        elecs = np.array(elecs) - 1 #adjust for indexing

        #Get minimum EI and check that all elecs 
        #are over threshold
        ei = np.amin(eis[nid,0], axis=1)
        main_elec = np.argmin(ei)
        main_elec_loc = getElecCoords512(main_elec)
        if np.all(np.array([ei[e] for e in elecs]) < thr):
            #check average distance from main elec
            elecs_locs = np.zeros((len(elecs),2))
            for ee,e in enumerate(elecs):
                elecs_locs[ee,:] = getElecCoords512(e)
            loc = np.mean(elecs_locs, axis=0)
            loc_dist = getLocDist(loc, main_elec_loc)
            if loc_dist > dist_thr: 
                output.append([nid, 'axon'])
              #  return 'axon'
            else: 
                output.append([nid, 'soma'])
              #  return 'soma'
        else:
            output.append([nid, 'far away'])
            #return 'far away'

    return output

def classifyElecSomaAxonDR(dr, neuron_ids, elecsarr):
    thr = -1
    dist_thr = 200


    eis = dr['datarun'][0,0]['ei']['eis'][0,0]
    cellIds = list(dr['datarun'][0,0]['cell_ids'][0,:])
    
    #Get neuron index
    output = []
    for nn, neuron_id in enumerate(neuron_ids):
        nid = cellIds.index(neuron_id)
        elecs = elecsarr[nn]
        elecs = np.array(elecs) - 1 #adjust for indexing

        #Get minimum EI and check that all elecs 
        #are over threshold
        ei = np.amin(eis[nid,0], axis=1)
        main_elec = np.argmin(ei)
        main_elec_loc = getElecCoords512(main_elec)
        if np.all(np.array([ei[e] for e in elecs]) < thr):
            #check average distance from main elec
            elecs_locs = np.zeros((len(elecs),2))
            for ee,e in enumerate(elecs):
                elecs_locs[ee,:] = getElecCoords512(e)
            loc = np.mean(elecs_locs, axis=0)
            loc_dist = getLocDist(loc, main_elec_loc)
            if loc_dist > dist_thr: 
                output.append([neuron_id, 'axon'])
              #  return 'axon'
            else: 
                output.append([neuron_id, 'soma'])
              #  return 'soma'
        else:
            output.append([neuron_id, 'far away'])
            #return 'far away'

    return output

def getEiVal(eis, cellIds, neuron_id):
        nid = cellIds.index(neuron_id)
        ei = np.amin(eis[nid,0], axis=1)
        main_min = np.min(ei)
        return main_min

def classifyElecSomaAxonEis(eis, cellIds, neuron_ids, elecsarr, elec_coords, adj_mat, distThr=100):
    thr = -1
    #dist_thr = 200
    dist_thr = 100
    
    #Get neuron index
    output = []
    for nn, neuron_id in enumerate(neuron_ids):
        nid = cellIds.index(neuron_id)
        elecs = elecsarr[nn]
        elecs = np.array(elecs) - 1 #adjust for indexing

        #Get minimum EI and check that all elecs 
        #are over threshold
        ei = np.amin(eis[nid,0], axis=1)
        main_elec = np.argmin(ei)
        #main_elec_loc = getElecCoords512(main_elec)
        main_elec_loc = eil.getSomaLoc2(eis, nid, elec_coords, adj_mat)
        #check average distance from main elec
        elecs_locs = np.zeros((len(elecs),2))
        for ee,e in enumerate(elecs):
            elecs_locs[ee,:] = eil.getElecCoords512(e)
        loc = np.mean(elecs_locs, axis=0)
        loc_dist = eil.getLocDist(loc, main_elec_loc)
        if np.all(np.array([ei[e] for e in elecs]) < thr):
            if loc_dist > dist_thr: 
                output.append([neuron_id, 'axon'])
              #  return 'axon'
            else: 
                output.append([neuron_id, 'soma'])
              #  return 'soma'
        else:
            if loc_dist > dist_thr: 
                output.append([neuron_id, 'far away'])
                #but give a warning
                print('Cell is close but EI is weak at electrode. Careful!')
            else:
                output.append([neuron_id, 'far away'])
            #return 'far away'

    return output

def classifyElecSomaAxonEisBE(eis, cellIds, neuron_ids, elecsarr,elec_coords, adj_mat, somathr=-8, distThr=100):
    thr = -1
    #dist_thr = 200
    dist_thr = distThr
    
    #Get neuron index
    output = []
    for nn, neuron_id in enumerate(neuron_ids):
        nid = cellIds.index(neuron_id)
        elecs = elecsarr[nn]
        elecs = np.array(elecs) - 1 #adjust for indexing

        #Get minimum EI and check that all elecs 
        #are over threshold
        ei = np.amin(eis[nid,0], axis=1)
        #main_elec = np.argmin(ei)
        #main_elec_loc = getElecCoords512(main_elec)
        #main_elec_loc = getSomaLoc(eis, nid, elec_coords, adj_mat, thr=somathr)
        main_elec_loc = eil.getSomaLoc2(eis, nid, elec_coords, adj_mat)
        if np.all(np.array([ei[e] for e in elecs]) < thr):
            #check average distance from main elec
            elecs_locs = np.zeros((len(elecs),2))
            for ee,e in enumerate(elecs):
                elecs_locs[ee,:] = getElecCoords512(e)
            loc = np.mean(elecs_locs, axis=0)
            loc_dist = getLocDist(loc, main_elec_loc)
            if loc_dist > dist_thr: 
                output.append([neuron_id, 'axon'])
              #  return 'axon'
            else: 
                output.append([neuron_id, 'soma'])
              #  return 'soma'
        else:
            output.append([neuron_id, 'far away'])
            #return 'far away'

    return output
def getElecsEIsDR(dr, neuron_ids, elecs):
    eis = dr['datarun'][0,0]['ei']['eis'][0,0]
    cellIds = list(dr['datarun'][0,0]['cell_ids'][0,:])
    output = []
    for nn, neuron_id in enumerate(neuron_ids):
        minoutput = []
        nid = cellIds.index(neuron_id)
        ei = np.min(eis[nid,0], axis=1)
        for e in elecs:
            minoutput.append(ei[e-1])
        output.append(minoutput)
    return output


def getElecRespAutoSummaryCells(elecRespAutoSummary, rowInd):
    return list(elecRespAutoSummary[rowInd,0].flatten())

def getElecRespAutoSummaryThrs(elecRespAutoSummary, rowInd):
    return list(elecRespAutoSummary[rowInd,1].flatten())

def getElecRespAutoSummaryCurves(elecRespAutoSummary, rowInd):
    return list(elecRespAutoSummary[rowInd,3].flatten())

def getElecRespAutoSummaryBundlethr(elecRespAutoSummary, rowInd):
    return list(elecRespAutoSummary[rowInd,5].flatten())[0]

def getElecRespAutoSummaryRSS(elecRespAutoSummary, rowInd):
    return list(elecRespAutoSummary[rowInd,2].flatten())

def getElecRespAutoSummaryFit(elecRespAutoSummary, rowInd):
    return list(elecRespAutoSummary[rowInd,6].flatten())

def getElecRespAutoSummaryIntercept(elecRespAutoSummary, rowInd):
    return list(elecRespAutoSummary[rowInd,7].flatten())

def getElecRespAutoSummaryIters(elecRespAutoSummary, rowInd):
    return list(elecRespAutoSummary[rowInd,8].flatten())

def getElecRespAutoSummaryTrials(elecRespAutoSummary, rowInd):
    return list(elecRespAutoSummary[rowInd,9].flatten())

def getEICellList(data_basename):
    dr = loadmat(data_basename + '_forpy.mat')
    eis = dr['datarun'][0,0]['ei']['eis'][0,0]
    cellIds = list(dr['datarun'][0,0]['cell_ids'][0,:])
    return cellIds

def getAutoPatElecs(dirnam):
    data = loadmat(dirnam)['elecResp'] 
    return data['stimInfo'][0,0]['electrodes'][0,0].flatten()

def getElecRespAutoSummaryElecs(elecRespAutoSummary, rowInd):
    try:
        return list(elecRespAutoSummary[rowInd,4].flatten())
        #return list(elecRespAutoSummary[rowInd,4][0][0].flatten())
    except IndexError:
        return [0]


def removeAmpsProbsDupl(amps_dup, probs_dup):
    amps = np.sort(np.unique(amps_dup))
    probs = []
    for a in amps:
        probs.append(np.sum(probs_dup[amps_dup==a])/np.sum(amps_dup==a))
    return amps, probs

def getReachableCells(ersumm_path, vispath,uniqueFlg=1):
    #Initialize output
    #Given elecRespSummary gives list of reachable cells
    ersumm = loadmat(ersumm_path)['elecRespAutoSummary']
    ns = []
    elecs = []
    bothelecs = []
    for p in ersumm:
        if np.any(p[0]) and np.any(p[1]): #If cells are stimulated with non-zero threshold
            #Get minimum threshold index of nonzero element
            minidx = np.argmin(p[1][p[1]!=0])
            #Get first one of the electrodes
            elec = list(p[-1][0][0][0])
            n = p[0][minidx][0]
            elecs.append(elec)
            foo = []
            for i in p[-1][0][0]:
                foo.append(i[0])
            bothelecs.append(foo)
            ns.append(n)
    if uniqueFlg: 
        #Make unique by only looking at first instance
        myns = set(ns)
        myelecs = []
        mybothelecs = []
        for n in myns:
            myelecs.append(elecs[ns.index(n)])
            mybothelecs.append(bothelecs[ns.index(n)])
        return (classifyElecSomaAxon(vispath, myns, myelecs), mybothelecs)
    else:
        return (classifyElecSomaAxon(vispath, ns, elecs), bothelecs)

def elecsToLocs(ersumm_path, vispath):

    dr = loadmat(vispath + '_forpy.mat')
    for p in ersumm:
        if np.any(p[0]) and np.any(p[1]): #If cells are stimulated with non-zero threshold
            neuron_ids = [n[0] for n in p[0]]
            nns = len(neuron_ids)
            locs.append(classifyElecSomaAxonDR(dr, neuron_ids, [[n[0] for n in p[-1][0][0] for i in range(nns)]]))

def loadEis(ei_path):
    eis = loadmat(ei_path)['datarun'][0,0]['ei']['eis'][0,0]
    return eis
            
def loadEiCellIds(ei_path):            
    cellIds = list(loadmat(ei_path)['datarun'][0,0]['cell_ids'][0,:])
    return cellIds

def loadEiCellTypes(ct_path,availableTypes=['On parasol','Off parasol','On midget','Off midget']):            
    foo = loadmat(ct_path)['celltypes']
    celltypes = []
    for i in foo[0,:]:
        grot = i[0].lower()
        if 'on' in grot and 'parasol' in grot: celltypes.append(availableTypes[0])
        elif 'off' in grot and 'parasol' in grot: celltypes.append(availableTypes[1])
        elif 'on' in grot and 'midget' in grot: celltypes.append(availableTypes[2])
        elif 'off' in grot and 'midget' in grot: celltypes.append(availableTypes[3])
        else: celltypes.append('other')

    return list(celltypes)

def getNumAxonSomaReachable(reachable_cells_non_unique):
    #Get list of unique cell ids
    cids = []
    for s in reachable_cells_non_unique:
        cids.append(s[0])

    #For each unique cell ID add to one of three bins
    numaxon = 0
    numsoma = 0
    numboth = 0
    axonlist = []
    somalist = []
    bothlist = []
    print(cids)
    for n in set(cids):
        foosoma = 0
        fooaxon = 0
        for s in reachable_cells_non_unique:
            if s[0] == n:
                if s[1] == 'soma': foosoma = 1
                elif s[1] == 'axon': fooaxon = 1
        if foosoma and fooaxon: 
            numboth +=1
            bothlist.append(n)
        elif foosoma: 
            numsoma += 1
            somalist.append(n)
        elif fooaxon: 
            numaxon += 1
            axonlist.append(n)
    return (numsoma, numaxon, numboth, somalist, axonlist, bothlist)

#Functions for manual ElecResp files
def getERei(ERpath):
    x = loadmat(ERpath)['elecResp']
    return x[0,0][0][0][0][3][0].split('/')[-1]

def getERfinalized(ERpath):
    x = loadmat(ERpath)['elecResp']
    try:
        return x[0,0][-1][0,0][4].flatten()
    except IndexError:
        return np.array([0])

def loadAutosortData(anbase,piece,aspath,bstem):
    erasumm = loadmat(anbase+piece+aspath+'/elecRespAutoSummary.mat')['elecRespAutoSummary']
    if bstem:
        bdir = glob.glob(anbase+piece+bstem+'*')
        bInds = loadmat(bdir[0]+ '/SafeZone_Indices.mat')['SafeZone_Indices'].flatten()-1
        for elec in range(512):
            row = getElecRespAutoSummaryCurves(erasumm,elec)
            if row:
                amps = row[0][1,:]
                break
        bThrs = amps[np.array([i if i < len(amps) else len(amps)-1 for i in bInds])]
    else:
        bThrs = []

def loadAutosortDataBcut(anbase,piece,aspath,bstem):
    erasumm = loadmat(anbase+piece+aspath+'/elecRespAutoSummaryBcut.mat')['elecRespAutoSummary']
    if bstem:
        bdir = glob.glob(anbase+piece+bstem+'*')
        bInds = loadmat(bdir[0]+ '/SafeZone_Indices.mat')['SafeZone_Indices'].flatten()-1
        for elec in range(512):
            row = getElecRespAutoSummaryCurves(erasumm,elec)
            if row:
                amps = row[0][1,:]
                break
        bThrs = amps[np.array([i if i < len(amps) else len(amps)-1 for i in bInds])]
    else:
        bThrs = []

    return erasumm,bThrs

def loadAutosortDataBcut2(anbase,piece,aspath,bpath):
    erasumm = loadmat(anbase+piece+aspath+'/elecRespAutoSummaryBcut.mat')['elecRespAutoSummary']
    bInds = loadmat(bpath+ '/SafeZone_Indices.mat')['SafeZone_Indices'].flatten()-1
    for elec in range(512):
        row = getElecRespAutoSummaryCurves(erasumm,elec)
        if row:
            amps = row[0][1,:]
            break
    bThrs = amps[np.array([i if i < len(amps) else len(amps)-1 for i in bInds])]

    return erasumm,bThrs

def loadAutosortDataSqueeze(piecepath,aspath,bstem,convertToArrays=False):
    erasumm = loadmat(piecepath+aspath+'/elecRespAutoSummary.mat', struct_as_record=False, squeeze_me=True)['elecRespAutoSummary']
    if convertToArrays:
        for row in range(erasumm.shape[0]):
            for col in range(erasumm.shape[1]):
                if col != 3:
                    if type(erasumm[row,col]) != np.ndarray: 
                        erasumm[row,col] = np.array([erasumm[row,col]])
                else: #amps/probs weirdness
                    if len(erasumm[row,col].shape) == 2:
                        erasumm[row,col] = np.array([erasumm[row,col]])
    if bstem:
        bdir = glob.glob(piecepath+bstem+'*')
        bInds = loadmat(bdir[0]+ '/SafeZone_Indices.mat')['SafeZone_Indices'].flatten()-1
        for elec in range(512):
            row = getElecRespAutoSummaryCurves(erasumm,elec)
            if row:
                try:
                    amps = row[0][1,:]
                    break
                except IndexError: #happens for unknown reason, perhaps one cell only rows?
                    continue
        bThrs = amps[np.array([i if i < len(amps) else len(amps)-1 for i in bInds])]
    else:
        bThrs = []

    return erasumm,bThrs

def loadManualSortData(anbase,piece,stimpath_manual):
    erasumm_manual = loadmat(anbase+piece+stimpath_manual+'/elecRespSummary.mat')['elecRespSummary']

    return erasumm_manual

def getActCurveType(probs):
    #if np.any(probs > 0.5):
    if np.argmax(probs > 0.5):
        #if probs[-1] < 0.5:
        if np.any(probs[np.argmax(probs > 0.5):] < 0.5):
            return 'updown'
        else:
            return 'normal'
    else:
        return 'no threshold'
