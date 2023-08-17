# Performe optimal sensorplacement 
# Author: Yi Meng Chan & Xicheng Wang
# --------------------------------------------------------------------------
#    Refer to" K. Manohar, et al "Data-driven sparse sensor placement 
#    for reconstruction: demonstrating the beneﬁts of exploiting known patterns",
#    IEEE Control Syst. Mag. 38 (3) 63–86 (2018)"
# --------------------------------------------------------------------------

import numpy as np
import numpy.matlib
import scipy.io
import os
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg


# Enter in the shape of the data here
Naxial  = 160
Nradial = 95

def trainTestIndex(Nsamples,testSplit_ratio):
    # The following test index was generated randomly using np.random.permutation(172)[0:22]
    testIndex = np.linspace(0,Nsamples-1, int(testSplit_ratio*Nsamples), dtype = int)
    # The remaining indices will be used for training
    trainIndex = np.ones(Nsamples, dtype=bool)
    trainIndex[testIndex] = False
    return trainIndex, testIndex

def getData():
    # Read all 3 groups together is too big, so read it separately 
    # path where stored the data
    dataLoc     = os.getcwd()
    folder_name = 'OptSensorPlacement'
    case_name   = ['PlaneJet2D_group1_1300cases.mat',\
                   'PlaneJet2D_group2_1120cases.mat',\
                   'PlaneJet2D_group3_1320cases.mat']
    
    Nsamples_tot = 3740
    count_sample = 0
    fields       = {}

    for ind in range(len(case_name)):
        filePath = f'{dataLoc[0:len(dataLoc)-len(folder_name)]}\TrainData\\{case_name[ind]}'
        mat      = scipy.io.loadmat(filePath)
        Nsamples = mat['T'].shape[1]
        print(f'Number of samples: {Nsamples}')
        if (ind == 0):
            # create a matrix that could cover 3 groups data together
            # create once
            fields['T']  = np.zeros((mat['T'][0,0].flatten().shape[0] , Nsamples_tot))
            fields['U']  = np.zeros((mat['U'][0,0].flatten().shape[0] , Nsamples_tot))
            xmesh        = mat['xmesh']
            ymesh        = mat['ymesh']
        for n in range(Nsamples):
            fields['T'][:, count_sample] = mat['T'][0, n].flatten()
            fields['U'][:, count_sample] = mat['U'][0, n].flatten()
            count_sample += 1

    return fields, xmesh, ymesh


def submean(f): # Subtract mean along samples
    nSamples = f.shape[1]
    fmean =  np.matlib.repmat(np.mean(f,axis=1).T,nSamples,1).T # along samples
    fsub  = f - fmean
    return fsub, fmean

def NMSE(ref,pred): # Evaulate normalized mean square error between reference and prediction
    res  = np.linalg.norm(ref-pred)**2
    tot  = np.linalg.norm(ref)**2
    nmse = res/tot
    return nmse

def optSensorPlacement(f,explained_variance_ratio_threshold): 
    # optSensorPlacement Provide optimal sensorplacement for given field
    U, S, _ = np.linalg.svd(f, full_matrices=False)
    explianed_vairance_ = (S**2)/(f.shape[1]-1)
    explianed_vairance_ratio = explianed_vairance_/explianed_vairance_.sum()
    cumVar = np.cumsum(explianed_vairance_ratio)

    # Find number of components to be extracted satifying the threshold variance 
    for i in range(len(S)):
        if cumVar[i] >= explained_variance_ratio_threshold:
            break
    numMoldes = i + 1

    _, _, sensor = scipy.linalg.qr(U[:,:numMoldes].T, mode='economic', pivoting=True)
    
    return U, sensor, numMoldes

if __name__ == "__main__":
    
    # prepare folders
    if os.path.exists('Figures') == False:
        os.mkdir('Figures')

    # Load data
    fields, xmesh, ymesh  = getData()
    Nsamples              = fields['T'].shape[1]

    # Subtract mean
    fields_sub   = {}
    fields_mean  = {}
    for var in ['T','U']:
        fields_sub[var], fields_mean[var] = submean(fields[var])   

    # index of training and test dataset
    trainIndex, testIndex = trainTestIndex(Nsamples,testSplit_ratio=0.10) 

    
    # Optimal sensor placement for U and T
    explained_variance_ratio_threshold = 0.999
    fields_svd_u    = {}
    fields_sensors  = {}
    fields_numModes = {}
    for var in ['T','U']:
        fields_svd_u[var], fields_sensors[var], fields_numModes[var]\
        = optSensorPlacement(fields_sub[var][:,trainIndex],explained_variance_ratio_threshold)
    
    nSensors       = 50   # number of sensors to be analyzed 
    nTrainSamples  = fields_sub[var].shape[1]
    nmse_tot       = {}
    # Evaluate performance for U and T
    for var in ['T','U']:
        nmse_tot[var] = np.zeros((nSensors))
        for k in range(nSensors): # Evaluate performance for each number of sensors
            sensors = fields_sensors[var][:k+1]
            nmse    = np.zeros((nTrainSamples))
            for i in range(nTrainSamples): # Evaluate performance for each sample
                snapind = i
                # a = (C Phi_r)^-1 y
                y = fields_sub[var][sensors,snapind]
                # reconstruct state in low-dimensonal space
                a     = np.linalg.pinv(fields_svd_u[var][sensors,:fields_numModes[var]])@y
                recon = np.dot(fields_svd_u[var][:,:fields_numModes[var]],a)
                pred  = fields_mean[var][:,snapind] + recon
                # Reference case
                ref   = fields[var][:,snapind]
                # error estimation
                nmse[i] = NMSE(ref,pred)
            nmse_tot[var][k] = nmse.sum()
    # NMSE vs number of sensors
    figure = plt.figure(figsize=(8,6))
    plt.plot(nmse_tot['T'],label='T')
    plt.plot(nmse_tot['U'],label='U')
    plt.yscale("log")
    plt.legend(loc='upper right')
    plt.xlabel("Number of sensors")
    plt.ylabel("NMES[-]")
    plt.savefig(f'Figures/NMSE_vs_numSensors_0999.png',dpi=300)
    plt.close

    # Visualize sensor position
    snapind    = 0
    plotField  = np.reshape(fields['T'][:,snapind], (Nradial, -1))
    Xind       = xmesh.flatten()
    Yind       = ymesh.flatten()
    sensorsT   = fields_sensors['T'][:fields_numModes['T']]
    sensorsU   = fields_sensors['U'][:fields_numModes['U']]
    
    figure = plt.figure(figsize=(8,6))
    plt.pcolormesh(xmesh,ymesh,plotField, cmap='jet',shading='nearest')
    plt.scatter(Xind[sensorsT], Yind[sensorsT], c='w',marker = 'o')
    plt.scatter(Xind[sensorsU], Yind[sensorsU], c='w',marker = '^')
    plt.axis('equal')
    plt.savefig(f'Figures/sensorPlacement_0999.png',dpi=300)
    plt.close() 
    
    # save index of optimal sensor position
    if os.path.exists('Data') == False:
        os.mkdir('Data')
    

    np.save('Data/optSensorPositionT_0999',sensorsT)
    np.save('Data/optSensorPositionU_0999',sensorsU)

          


