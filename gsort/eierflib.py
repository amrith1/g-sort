import gsort.eilib as eil
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
pd.set_option('display.max_columns', None)
import pickle
import gsort.stimlib as st
import random
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import gsort.old_labview_data_reader as oldr
from scipy import stats
import gsort.post_processing_utils as gpa #jeff

#GLOBALS----------------------------------------
#define curve fits
eierfParams = {'soma':(22,.4,30),'axon':(6.5,.4,10),'mixed':(6.5,.4,10)}
eierfParamsV = {'soma':(22*3.6+20,.4,30*3.6*.75),'axon':(6.5*3.6,.4,10*3.6*1.1),'mixed':(6.5,.4,10)}
eierfParamsMidgetV = {'soma':(22*3.6+15,.3*.76,7),'axon':(6.5*3.6,.28,3*3),'mixed':(6.5,.4,10)}
slopethrParams = {'axon': (4.9075161584664855, 8.728166702074772, .6), 'soma': (3.9460637786474644, 2.2495220647477385, .19),
                 'mixed': (4.9075161584664855, 8.728166702074772, .6)}

eiLThr = {'soma':-eierfParams['soma'][-1]-4, 'axon':-eierfParams['axon'][-1]-3, 'mixed':-eierfParams['mixed'][-1]-3} #for fair prediction
eiLThrMidget = {'soma': -20, 'axon': -7, 'mixed': -7}

outLStdThr=0.1
ll_std_lims={'soma':5,'axon':.5,'mixed':.5,'dendrite':.5,'error':.5}

#paths = eil.loadPaths()
#coords = eil.loadCoords512()

#GENERAL----------------------------------------
def sigmoid(x,slope,thr):
    return 1/(1+np.exp(-slope*(x-thr)))

def eiErfCurve(x,a,b,c):
    return a/(x-c) + b

def slopeThrCurve(x,a,b,c):
    return a/(x-c) + b

#PREDICTION----------------------------------------
def slopeFlg(row):
    if row['dcompartment'] == 'soma':
        if row['slope'] > 25:
            return False
        else:
            return True
    else:
        return True

def linearFit(xRange,a,b):
    return a*xRange + b

def getEiErfThr(row,eierfParams):
    buffer = 4 if row['dcompartment'] == 'soma' else 2
    if row['eiVal'] < -eierfParams[row['dcompartment']][2]-buffer: #CHANGE??
        return eiErfCurve(np.abs(row['eiVal']),*eierfParams[row['dcompartment']])
    else:
        return 0

def getLastProbBthr(row):
    try:
        idx = np.where(row['amps']==row['bthr'])[0][0]
        if idx > 0:
            return np.max(row['probs'][:idx])
        else:
            return 0
    except IndexError: #if amps is somehow broken
        return 0

def somaDeviation2(dist,width):
    #params = (0.014597096150856737, -0.4284233245362835)
    params = (0.017890132615959405, -0.5710449693724807) #SLIGHTLY MODIFIED (FIT ON HIGH EI + HIGH NOISE POINTS ONLY)
    return params[0]*(dist*(width**0.5))+params[1]

def aisDist(dist,width):
    return dist*(width**0.5)
    
def getSomaDeviatedThr(row):
    if row['dcompartment']=='soma':
        predThr_0 = eiErfCurve(np.abs(row['eiVal']),*eierfParams['soma'])
        return predThr_0 + somaDeviation2(row['hotspot_distance'],row['soma_width_axon'])*predThr_0
    else:
        return eiErfCurve(np.abs(row['eiVal']),*eierfParams[row['dcompartment']])
    
#geometry
def get_soma_coord(row,df,coords): #must compute first
    fullCellDf_soma = df[(df['piece-cell']==row['piece-cell']) & (df['compartment'] == 'soma')]
    somaPos = eil.getSomaLocThr(fullCellDf_soma['elecInd'].astype(int), fullCellDf_soma['eiVal'],coords)
    return somaPos
    
def get_axon_coord(row,df,coords):
    #if row['piece'] in pieces and row['celltype'] in celltypes: #to save time
    #minDist = 1e8; axCoord = (0,0); uppBound=1.8; lowBound=0.05 #modified from default bounds
    minDist = 1e8; axCoord = np.array([0,0]); uppBound=1.4; lowBound=0.05 #modified from default bounds
    fullCellDf = df[df['piece-cell']==row['piece-cell']]
    #round 1, for most cells
    for index,row2 in fullCellDf.iterrows():#get pos of nearest axon
        if eil.axonorsomaRatio(row2['waveform'],uppBound=uppBound,lowBound=lowBound) == 'axon':
            #if 230 < row2['distance'] < minDist: #230 is arbitrary
            if 190 < row2['distance'] < minDist and row2['eiVal']<-10: #190 is arbitrary
                minDist = row2['distance']
                axCoord = coords[int(row2['elecInd']),:]
    #round 2 (catches small cells)
    if np.all(axCoord == np.array([0,0])):
        for index,row2 in fullCellDf.iterrows():#get pos of nearest axon
            if eil.axonorsomaRatio(row2['waveform'],uppBound=uppBound,lowBound=lowBound) == 'axon':
                #if 230 < row2['distance'] < minDist: #230 is arbitrary
                #print(row2['eiVal'])
                if 190 < row2['distance'] < minDist and row2['eiVal']<-4: 
                #if 190 < row2['distance'] < minDist: #230 is arbitrary
                    minDist = row2['distance']
                    axCoord = coords[int(row2['elecInd']),:]
    #round 3 (catches small cells)
    if np.all(axCoord == np.array([0,0])):
        for index,row2 in fullCellDf.iterrows():#get pos of nearest axon
            if eil.axonorsomaRatio(row2['waveform'],uppBound=uppBound,lowBound=lowBound) == 'axon':
                #if 230 < row2['distance'] < minDist: #230 is arbitrary
                #print(row2['eiVal'])
                if 190 < row2['distance'] < minDist and row2['eiVal']<-2: 
                #if 190 < row2['distance'] < minDist: #230 is arbitrary
                    minDist = row2['distance']
                    axCoord = coords[int(row2['elecInd']),:]
    return axCoord

def get_axon_coord_raw_30um(ei,coords,eiThr=-2):
    #if row['piece'] in pieces and row['celltype'] in celltypes: #to save time
    #minDist = 1e8; axCoord = (0,0); uppBound=1.8; lowBound=0.05 #modified from default bounds
    minDist = 1e8; axCoord = np.array([0,0]); uppBound=1.4; lowBound=0.05 #modified from default bounds

    eiFlat = np.min(ei,axis=1)
    eiVals = eiFlat[eiFlat<eiThr]
    elecInds = np.where(eiFlat<eiThr)[0]
    waveforms = [ei[ind,:].flatten() for ind in elecInds]

    #specific to 519
    cnt = 0
    for waveform in waveforms:
        if eil.axonorsomaRatio(waveform,uppBound=uppBound,lowBound=lowBound) == 'axon':
            cnt +=1
    if cnt < 7: return (0,0) #RETURN


    soma_pos,_,_ = eil.getSomaLocWaveRatioLean(ei,coords,uppBound=1.6,lowBound=0.05,eithr=eiThr)

    #round 1, for most cells
    for waveform,elecInd,eiVal in zip(waveforms,elecInds,eiVals):#get pos of nearest axon
        if eil.axonorsomaRatio(waveform,uppBound=uppBound,lowBound=lowBound) == 'axon':
            if 190 < eil.getLocDist(soma_pos,coords[elecInd,:]) < minDist and eiVal<-10: #190 is arbitrary
                minDist = eil.getLocDist(soma_pos,coords[elecInd,:])
                axCoord = coords[elecInd,:]
    #round 2 (catches small cells)
    if np.all(axCoord == np.array([0,0])):
        for waveform,elecInd,eiVal in zip(waveforms,elecInds,eiVals):#get pos of nearest axon
            if eil.axonorsomaRatio(waveform,uppBound=uppBound,lowBound=lowBound) == 'axon':
                if 190 < eil.getLocDist(soma_pos,coords[elecInd,:]) < minDist and eiVal<-4: #190 is arbitrary
                    minDist = eil.getLocDist(soma_pos,coords[elecInd,:])
                    axCoord = coords[elecInd,:]
    #round 3 (catches small cells)
    if np.all(axCoord == np.array([0,0])):
        for waveform,elecInd,eiVal in zip(waveforms,elecInds,eiVals):#get pos of nearest axon
            if eil.axonorsomaRatio(waveform,uppBound=uppBound,lowBound=lowBound) == 'axon':
                if 190 < eil.getLocDist(soma_pos,coords[elecInd,:]) < minDist and eiVal<-4: #190 is arbitrary
                    minDist = eil.getLocDist(soma_pos,coords[elecInd,:])
                    axCoord = coords[elecInd,:]
    return axCoord

#INSPECTION----------------------------------------

def inspect_plot(row,df,coords,labels=['']):
    fullCellDf = get_cell_df(df,row)
    
    fig,ax = plt.subplots(1,4,figsize=(24*4/5,4))

    #plot ei + stim elec
    eil.plotEiColsThr(fullCellDf['elecInd'].astype(int), fullCellDf['eiVal'], fullCellDf['waveform'], ax[0],
                      coords, size_fac=-20,alpha=.7,lowBound=0.1)
    ax[0].scatter(*coords[int(row['elecInd']),:],marker='x',s=120,c='k')
    fullCellDf = fullCellDf[fullCellDf['compartment'] == 'soma']
    ax[0].scatter(*eil.getSomaLocThr(fullCellDf['elecInd'].astype(int), fullCellDf['eiVal'],coords),marker='*',s=150,c='r',alpha=.7)
    ##axon position
    try:
        ax[0].scatter(*row['axon_position'],marker='x',s=100,lw=3,c='b')
    except TypeError:
        print('no axon position')
        #pass
    #ax[0].scatter(*row['axon_position_x'],marker='^',s=100,lw=2,c='b')
    #ax[0].scatter(*row['axon_position_y'],marker='^',s=100,lw=2,c='r')

    #plot waveform
    ax[1].plot(row['waveform'])
    ax[1].set_xlabel('time (samples)')
    ax[1].set_ylabel('voltage (DAC)')

    #plot act curve
    xRange = np.linspace(0,5)
    colors = [eil.corrColDict['b'],eil.corrColDict['r'],eil.corrColDict['g']]
    for ll,label in enumerate(labels):
        if type(row['probs_gsort']) == np.ndarray or type(row['probs_gsort']) == list:
            #if row['probs'+label] and not np.all(np.isnan(row['probs'+label])):
            if not np.all(np.isnan(row['probs'+label])):
                labNam = label[1:]
                if len(row['probs'+label]) < len(row['amps'+label]):
                    ampsFoo = row['amps'+label][:len(row['probs'+label])]
                else:
                    ampsFoo = row['amps'+label]
                ax[2].scatter(ampsFoo,row['probs'+label],c=colors[ll],alpha=.6)
                ax[2].plot(xRange,sigmoid(xRange,4*row['slope'+label],row['threshold'+label]),label=labNam,c=colors[ll])
    #ax[2].plot(xRange,sigmoid(xRange,row['pred_slope'],row['ei_erf_thr']),c=eil.corrColDict['g'],label='predicted')
    if 'deviated_pred_thr' in row.keys():
        if row['deviated_pred_thr']:
            ax[2].plot(xRange,sigmoid(xRange,row['deviated_pred_slope'],row['deviated_pred_thr']),c=eil.corrColDict['k'],
                       label='predicted',linestyle='dashed')
    ax[2].axvline(x=row['bthr'],linestyle='dashed',c='r',lw=1.5)
    ax[2].set_xlabel('stim amp (\u03BCA)')
    ax[2].set_ylabel('spike prob')
    ax[2].set_xlim([0,4.5])
    ax[2].legend()

    #add control sigmoid #TOGGLE
    #thr = 1 #blind default
    NUM_SIGMAS = 1.7
    rf = row['rf']
    try:
        tilt = np.array(rf.rot) * (180 / np.pi) * -1
        fit = Ellipse(xy = (rf.center_x,rf.center_y), width= (NUM_SIGMAS * rf.std_x),
              height= (NUM_SIGMAS * rf.std_y),angle=tilt,lw=2.3)
        ax[3].add_artist(fit)
        fit.set_facecolor('None')
        fit.set_edgecolor(eil.corrColDict['b'])
        try:
            ax[3].set_xlim([rf.center_x-3, rf.center_x+3])
            ax[3].set_ylim([rf.center_y-3, rf.center_y+3])
        except ValueError:
            pass
    except AttributeError:
        pass
    

    #templates at that electrode and neighboring (default gonzalo setting)
    try:
        mainElec = fullCellDf['elecInd'].loc[fullCellDf['eiVal'].idxmin()] + 1
    except ValueError: #no soma elecs
        mainElec = 'none'
        #get rss
    #get rss
    #rss = np.sum((sigmoid(row['amps'],row['slope'],row['threshold']) - row['probs'])**2)
    fig.suptitle(row['piece'] + ' ' + row['celltype'] + ' ' + str(row['cell']) + ' elec: ' + str(int(row['elecInd']+1)) +\
                 ' axonal elecs: ' + str(row['naxon']) + ' main elec: ' + str(mainElec) +\
                 ' compts: r-'+row['compartment']+', d-'+row['dcompartment'] + ' n:'+str(row['noise']))
    
    return fig,ax,mainElec

def get_cell_df(df,row):
    df1 = df[(df['cell'] == row['cell']) & (df['piece'] == row['piece']) & (df['ei_dataset']==row['ei_dataset'])]
    #df1 = df[(df['piece-cell-electrode-ei'] == row['piece-cell-electrode-ei'])]
    return df1

#GSORT CHECKS----------------------------------------
def showClusters(row,ampInd,dfM,coords,gsortLabs=['_mod_elec_noise_error']):
    
    numCol = 3
    
    for gsLab in gsortLabs:
        print(gsLab)
        
        #rec_elecInds = np.array(row['electrode_list'+gsLab][ampInd])
        rec_elecInds = np.array(row['electrode_list'+gsLab])
        fig,ax = plt.subplots(2*int(np.ceil(len(rec_elecInds)/numCol)),
                              numCol,figsize=(20*1.2,4*.7*int(np.ceil(len(rec_elecInds)/3))*2*1.2*1.5))

        pattern_no = int(row['elecInd']+1)
        analysis_path = '/Volumes/Analysis/'+row['piece']+row['stim_dataset']+'/'
        raw_traces = oldr.get_oldlabview_pp_data(analysis_path, pattern_no, ampInd)
        #clList = np.array(row['clusters_list'][ampInd]); clusters = np.sort(np.unique(clList))
        clList = np.array(row['initial_clustering_with_virtual'+gsLab][ampInd]); clusters = np.sort(np.unique(clList))
        norm = mpl.colors.Normalize(vmin=0, vmax=len(clusters)-1, clip=True); mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_earth)
        for cc,cluster in enumerate(clusters):
            #if cluster not in np.array(list(row['cell_in_cluster_list'][ampInd].values())).flatten():
            #if cluster not in [x for i in list(row['cell_in_cluster_list'][ampInd].values()) for x in i]:
            cell_ids = []
            if cluster not in [x for i in list(row['cell_in_clusters'+gsLab][ampInd].values()) for x in i]:
                cell_id = 0
                cell_ids.append(0)
            else:
                #for key,value in row['cell_in_cluster_list'][ampInd].items():
                for key,value in row['cell_in_clusters'+gsLab][ampInd].items():
                    if cluster in value:
                        cell_id = int(key)
                        cell_ids.append(int(key))
            trials = np.where(clList == cluster)[0]
            #ax[1].plot(raw_traces[trials,rec_elecInd,:].T,label='cluster '+str(cluster),c=mapper.to_rgba(cc))
            for rr,rec_elecInd in enumerate(rec_elecInds):
                ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].plot(raw_traces[trials,rec_elecInd,:55].T,
                              #label='cell '+str(cell_id) + ' ' + str(cluster) + ' #:' + str(len(trials)),c="C"+str(cc%10),lw=.7)
                              label='cell '+str(cell_ids) + ' ' + str(cluster) + ' #:' + str(len(trials)),c="C"+str(cc%10),lw=.7)
                #plot waveform
                try:
                    ax[int(np.floor(rr/numCol)*2+1),int(rr%numCol)].plot(dfM[(dfM['piece']==row['piece']) & (dfM['cell']==cell_id) &\
                                      #(dfM['elecInd']==rec_elecInd)].iloc[0]['waveform'],c=mapper.to_rgba(cc),
                                      (dfM['elecInd']==rec_elecInd)].iloc[0]['waveform'],c="C"+str(cc%10),
                                  label='cell '+str(cell_id) + ' ' + str(cluster))
                                  #label='cell '+str(cell_ids) + ' ' + str(cluster))

                    ax[int(np.floor(rr/numCol)*2+1),int(rr%numCol)].plot(dfM[(dfM['piece']==row['piece']) & (dfM['cell']==row['cell']) &\
                                      (dfM['elecInd']==rec_elecInd)].iloc[0]['waveform'],c='b',linestyle='dashed',
                                  #label='cell '+str(row['cell']) + ' autosort chosen')
                                  label='cell '+str(row['cell']) + ' A.C.')
                except IndexError:
                    pass

        for rr,rec_elecInd in enumerate(rec_elecInds):
            handles, labels = ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].get_legend_handles_labels(); by_label = dict(zip(labels, handles))
            ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].legend(by_label.values(), by_label.keys(),fontsize=5.5*2)
            ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].set_title('elecInd: ' + str(rec_elecInd) + ' ampInd: ' + str(ampInd),fontsize=20)
            handles, labels = ax[int(np.floor(rr/numCol)*2+1),int(rr%numCol)].get_legend_handles_labels(); by_label = dict(zip(labels, handles))
            ax[int(np.floor(rr/numCol)*2+1),int(rr%numCol)].legend(by_label.values(), by_label.keys(),fontsize=7*2,loc='lower right')

            hrange = np.abs(ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].get_ylim()[0]-\
                            ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].get_ylim()[1])/2
            ax[int(np.floor(rr/numCol)*2+1),int(rr%numCol)].set_ylim([0-hrange,0+hrange])
            ax[int(np.floor(rr/numCol)*2+1),int(rr%numCol)].set_xlim([0,54])
        
    #return most common cell (for misassignments)
    cellToCluster = []
    for cluster in row['initial_clustering_with_virtual'+gsLab][ampInd]:
        for cell in row['cell_in_clusters'+gsLab][ampInd].keys():
            if cluster in row['cell_in_clusters'+gsLab][ampInd][cell]:
                cellToCluster.append(cell)
    mostCommonCell = stats.mode(cellToCluster).mode[0]
    
    
    return fig,ax,mostCommonCell

def plotTopElecs(row,df,ampInd,targetCell,coords,gsortLabs=['_mod_elec_noise_error']):
    numCol = 3
    ##get elecs
    fullCellDf = df[(df['piece']==row['piece']) & (df['cell']==targetCell)]
    fullCellDf = fullCellDf.reset_index()
    rec_elecInds = fullCellDf.iloc[np.argsort(fullCellDf['eiVal'].values)[:3]]['elecInd']
    
    for gsLab in gsortLabs:
        print(gsLab)
        
        fig,ax = plt.subplots(2*int(np.ceil(len(rec_elecInds)/numCol)),
                              numCol,figsize=(20*1.2*.8,4*.7*int(np.ceil(len(rec_elecInds)))*2*.8*.8))

        pattern_no = int(row['elecInd']+1)
        analysis_path = '/Volumes/Analysis/'+row['piece']+row['stim_dataset']+'/'
        raw_traces = oldr.get_oldlabview_pp_data(analysis_path, pattern_no, ampInd)
        
        for rr,rec_elecInd in enumerate(rec_elecInds):
                ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].plot(raw_traces[:,int(rec_elecInd),:55].T,
                              c="C"+str(rr%10),lw=.7)
                ax[int(np.floor(rr/numCol)*2+1),int(rr%numCol)].plot(df[(df['piece']==row['piece']) & (df['cell']==targetCell) &\
                                  (df['elecInd']==int(rec_elecInd))].iloc[0]['waveform'],c="C"+str(rr%10))
                
        for rr,rec_elecInd in enumerate(rec_elecInds):
            ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].set_title('cell: ' + str(targetCell) +\
                                                                    ' elecInd: ' + str(int(rec_elecInd)) +\
                                                                    ' ampInd: ' + str(ampInd),fontsize=20)
            hrange = np.abs(ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].get_ylim()[0]-\
                            ax[int(np.floor(rr/numCol))*2,int(rr%numCol)].get_ylim()[1])/2
            ax[int(np.floor(rr/numCol)*2+1),int(rr%numCol)].set_ylim([0-hrange,0+hrange])
            ax[int(np.floor(rr/numCol)*2+1),int(rr%numCol)].set_xlim([0,54])
            
        return fig,ax

def showCell(row,cells,df,coords):
    fig,ax = plt.subplots(1,1,figsize=(9*2,6*2))
    titstr = ''
    for cell in cells:
        fullCellDf = df[(df['piece']==row['piece']) & (df['cell']==cell)]
        ##get main elec
        fullCellDf = fullCellDf.reset_index()
        mainElec = fullCellDf.iloc[fullCellDf['eiVal'].idxmin()]['elecInd']
        #plot ei + stim elec
        #eil.plotEiColsThr(fullCellDf['elecInd'], fullCellDf['eiVal'], fullCellDf['waveform'], ax,
        #                  coords, size_fac=-20,alpha=.7,lowBound=0.1)
        eil.plotEiTextThr(fullCellDf, ax, coords, size_fac=-12,ei_color='',
                                  alpha=.6,label=None,plotArray=True,foreground='w',squared=False,
                                  fontsize=24,linewidth=5)
        fullCellDfSoma = fullCellDf[fullCellDf['compartment'] == 'soma']
        ax.scatter(*eil.getSomaLocThr(fullCellDfSoma['elecInd'], fullCellDfSoma['eiVal'],coords),marker='*',s=150,c='r',alpha=.7)
        titstr = titstr + 'cell: ' + str(cell) + ' mainElec: ' + str(mainElec) + ' ' + fullCellDf['celltype'].iloc[0] + ' '
        
    ax.scatter(*coords[int(row['elecInd']),:],marker='x',s=120,c='k')
    ax.set_title(titstr)
    
    return fig,ax


def showSigmoids(row,ax,indFS=0,asLab='',gsorts=['_mod_elec_noise_error']):
    xs = np.linspace(0,5)
    for ampKey,probKey,thrKey,slKey,label,color in zip(['amps']+['amps'+i for i in gsorts],
                                                       ['probs']+['probs_filtered'+i for i in gsorts],
                                                       ['threshold']+['threshold_smart_filtered'+i for i in gsorts],
                                                       ['slope']+['slope_smart_filtered'+i for i in gsorts],
                                                       ['autosort']+['gsort_filtered'+i for i in gsorts],
                                                       ['b','y','g','y','c','k']):
        if np.any(row[probKey]):
            print(probKey)
            ampsFoo = row[ampKey][:len(row[probKey])]
            ax.scatter(ampsFoo,row[probKey],label=label,c=eil.corrColDict[color],s=190)
            ax.plot(xs,sigmoid(xs,row[slKey]*4,row[thrKey]),c=eil.corrColDict[color])
            #plot text
            if indFS:
                for ii,(x,y) in enumerate(zip(row[ampKey],row[probKey])):
                    ax.text(x,y,str(ii),fontsize=indFS)
    ##bundle
    ax.axvline(row['bthr'],linestyle='dashed',lw=1.4,c='k')
    ax.legend(fontsize=22)
    
    ax.set_title(row['piece-cell-electrode-ei'],fontsize=24)

def inspect_cell_raw(ei,coords,axon_pos=0,eiThr=-2,ax=None):

    if not ax:
        fig,ax = plt.subplots(1,1,figsize=(6,4))
        axFlg = True
    else:
        axFlg = False

    eiFlat = np.min(ei,axis=1)
    eiVals = eiFlat[eiFlat<eiThr]
    elecInds = np.where(eiFlat<eiThr)[0]
    waveforms = [ei[ind,:].flatten() for ind in elecInds]

    #plot ei + stim elec
    eil.plotEiColsThr(elecInds, eiVals, waveforms, ax,
                      coords, size_fac=-20,alpha=.7,lowBound=0.1)
    soma_pos,_,_ = eil.getSomaLocWaveRatioLean(ei,coords,uppBound=1.6,lowBound=0.05,eithr=eiThr)
    ax.scatter(*soma_pos,marker='*',s=150,c='r',alpha=.7)
    ##axon position
    if type(axon_pos) != int:
        ax.scatter(*axon_pos,marker='x',s=100,lw=3,c='b')

    if axFlg:
        return fig,ax,soma_pos
    else:
        return 0,0,soma_pos

def smart_fit(amps,probs,pointsToCut=2):
    ampsFoo = amps[:len(probs)] if len(probs) < len(amps) else amps
    probsFoo = probs[:len(amps)] if len(amps) < len(probs) else probs
    threshJ,slopeJ = gpa.infer_sigmoid(np.array(probsFoo)[pointsToCut:],np.array(ampsFoo)[pointsToCut:],mono_threshold=0.19,
                                       noise_limit=0.10, kind='add')
    return threshJ,slopeJ,ampsFoo,probsFoo

def prettify_fonts(ax,axisTickSize=12,axisTextSize=14,titleTextSize=15):
    ax.tick_params(axis='both', which='major', labelsize=axisTickSize)
    ax.xaxis.get_label().set_fontsize(axisTextSize)
    ax.yaxis.get_label().set_fontsize(axisTextSize)
    ax.title.set_fontsize(titleTextSize)
