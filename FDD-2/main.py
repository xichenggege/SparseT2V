# Note PCA is performed with sklearn package
# Neural network is performed with tensorflow
# Author: Yi Meng Chan & Xicheng Wang

# generalRegressionFramework is a private github repo found at: YimengChankth/generalRegressionFramework

import generalRegressionFramework.PCA as PCA
import generalRegressionFramework.DNN as DNN

import numpy as np
import scipy.io
import os
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp


# Enter in the shape of the data here
Naxial  = 160
Nradial = 95
# Train_epoch_1 = 5000
Train_epoch_2 = 10000

def trainTestIndex(Nsamples,testSplit_ratio):
    # The following test index was generated randomly using np.random.permutation(172)[0:22]
    testIndex = np.linspace(0,Nsamples-1, int(testSplit_ratio*Nsamples), dtype = int)
    # The remaining indices will be used for training
    trainIndex = np.ones(Nsamples, dtype=bool)
    trainIndex[testIndex] = False
    return trainIndex, testIndex

def sensorSampling(Tfield, showTCgrid, savefolder='Figures'):
    '''
    put showTCgrid to True to save a plot of the TC placement as determined by sensorPlacements
    '''
    # Tdata = np.reshape(Tdata, (30,-1))
    # plt.imshow(Tdata.T, cmap='viridis')
    
    sensorsIndicator = sensorPlacements
    # plot on representative flow

    if showTCgrid == True:
        T = np.reshape(Tfield[:, 0], (Nradial, -1))
        plt.contourf(xmesh,ymesh,T, cmap='viridis')
        plt.scatter(xmesh.flatten()[sensorPlacements], ymesh.flatten()[sensorPlacements], c='r')
        plt.axis('equal')
        plt.savefig(f'{savefolder}/sensorPlacement.png',dpi=300)
        plt.close() 

    Tsensor = Tfield[sensorsIndicator,:]

    return Tsensor

def getData():
    # Read all 3 groups together is too big, so read it separately 
    # path where stored the data
    dataLoc     = os.getcwd()
    folder_name = 'FDD-2'
    case_name   = ['PlaneJet2D_group1_1300cases.mat',\
                   'PlaneJet2D_group2_1120cases.mat',\
                   'PlaneJet2D_group3_1320cases.mat']
    
    Nsamples_tot = 3740
    count_sample = 0
    fields       = {}

    for ind in range(len(case_name)):
        filePath = f'{dataLoc[0:len(dataLoc)-len(folder_name)]}TrainData\\{case_name[ind]}'
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

def plotPCAComponents(pca : PCA.PCA, savefolder):
    '''Plot the PCA components and save them to the folder: PCAComponents/
    '''
    if os.path.exists(savefolder) == False:
        os.mkdir(savefolder)
        print(f'Created new folder at: {savefolder}')  
    components = pca.components_

    for n in range(pca.n_components_):
        figure = plt.figure(figsize=(8,6))
        data = components[n,:].flatten()
        # Reshape to 30 x 100
        data = np.reshape(data, (Nradial,-1))
        # Make the data appear similar to the model geometry. Flow moving downwards from the top
        plt.contourf(xmesh,ymesh,data, cmap='viridis')
        plt.title(f'PCA component {n+1}')
        plt.axis('equal')
        plt.savefig(f'{savefolder}/component_{n+1}', dpi=300)
        plt.close(figure)
        print(f'Saved component {n+1} to {savefolder}/component_{n+1}.png')

    # plot the cumulative explained variance ratio
    figure = plt.figure(figsize=(8,6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('# components')
    plt.title('Cumulative explained variance ratio')
    plt.savefig(f'{savefolder}/cumulativeExplainedVarianceRatio.png', dpi=300)
    plt.close()
    pass

def saveToMatlab_pca(pca, savename):
    '''Save test data back to matlab 
    '''
    results = dict()
    components = pca.components_

    reshapedComponents = np.zeros((Naxial, Nradial,pca.n_components_))

    for n in range(pca.n_components_):
        # figure = plt.figure(figsize=(8,6))
        data = components[n,:].flatten()
        # Reshape
        data = np.reshape(data, (Nradial,-1))
        reshapedComponents[:,:,n] = data.T

    results['pcaComponents'] = reshapedComponents
    results['explained_variance_ratio_n'] = pca.explained_variance_ratio_
    sp.io.savemat(savename, results)
    pass 

def saveToMatlab_dnn(caseIndex, sensorIndexPosition, reference, predict, hist, savename):
    '''Save test data back to matlab 
    '''
    print(f'Saving test data to a MATLAB file')
    results = dict()
    results['caseIndex']  = caseIndex
    results['predict']    = predict
    results['reference']  = reference
    results['sensorIndexPosition'] = sensorIndexPosition
    results['Nsensors'] = sensorIndexPosition.shape[0]
    results['hist_trainloss']     = hist.history['loss']
    results['hist_valoss']        = hist.history['val_loss']
    
    sp.io.savemat(savename, results)
    pass  

def plotTrainHistory(history,savefolder):
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.yscale('log')
    plt.savefig(f'{savefolder}/trainHisotry.png',dpi=300)
    plt.close()
    pass

def plotTest(ref,pred,plotSensor,savefolder):
    numPlot = ref.shape[1]
    for count in range(numPlot):
        print(f'Printing test index {count+1} reconstruction')
        fig, axs = plt.subplots(2,1,figsize=(6,8))
        vmin = min(ref[:, count])
        vmax = max(ref[:, count])
        # refimshow = axs[0].imshow(np.reshape(ref[:, count], (Nradial,-1)).T, cmap='jet', vmin = vmin, vmax = vmax)
        refimshow = axs[0].contourf(xmesh, ymesh, ref[:,count].reshape(Nradial,Naxial), \
                    cmap='jet', vmin = vmin, vmax = vmax)
        axs[0].set_title('Ref')
        axs[0].axis('equal')
        # Plot sensors
        if plotSensor == True:
            axs[0].scatter(xmesh[1,sensorPlacements[:, 1]], ymesh[sensorPlacements[:, 0],1], c='r', s=5)

        axs[1].contourf(xmesh, ymesh, pred[:,count].reshape(Nradial,Naxial), \
                       cmap='jet', vmin = vmin, vmax = vmax)
        axs[1].set_title('Predict')
        axs[1].axis('equal')
        formatter = matplotlib.ticker.StrMethodFormatter('{x:.1f}')
        fig.colorbar(refimshow, ax=axs[:], location='right', shrink=0.6,  format=formatter)
        fig.suptitle(f'Test index: {testIndex[count]}')
        # fig.tight_layout()
        plt.savefig(f'{savefolder}/test{testIndex[count]}.png',dpi=300)
        plt.close()   
    pass

def submean(f): # Subtract mean along samples
    nSamples = f.shape[1]
    fmean = np.matlib.repmat(np.mean(f,axis=1).T,nSamples,1).T  # along samples
    fsub  = f - fmean
    return fsub, fmean
    
if __name__ == "__main__":
    
    # Load data
    fields, xmesh, ymesh  = getData()
    Nsamples              = fields['T'].shape[1]
    sensorPlacements      = np.load('optSensorPositionT_0999.npy')

    # index of training and test dataset
    trainIndex, testIndex = trainTestIndex(Nsamples,testSplit_ratio=0.10)   

    # Subtract mean
    fields_sub   = {}
    fields_mean  = {}
    for var in ['T','U']:
        fields_sub[var], fields_mean[var] = submean(fields[var])  

    TS   = sensorSampling(fields_sub['T'][:, trainIndex], showTCgrid=True, savefolder='Figures/') # T sparse space
    # -------------------------------------------------------------------------
    # Run n times for each setup
    # -------------------------------------------------------------------------
    num_run = 1
    for runIndex in range(num_run):
        # Create folder to save each run results
        if os.path.exists(f'run{runIndex}') == False:
            os.mkdir(f'run{runIndex}')
                # prepare folders
        if os.path.exists(f'run{runIndex}/pred') == False:
            os.mkdir(f'run{runIndex}/pred')
   
        # -------------------------------------------------------------------------
        # Part 1: U full space to U latent space by PCA
        # -------------------------------------------------------------------------
        # PCA with explained variance ratio = 0.999 
        # Set up PCA and train with explained variance ratio 0.999. normalize == True tends to `flatten` the importance of individual meshes. 

        d_U = PCA.PCA(max_components=100, min_components=1, explained_variance_ratio_threshold=0.999, normalize=False)
        d_U.train(train = fields_sub['U'][:, trainIndex])       # train PCA
        print(f'PCA # of components: {d_U.n_components_} .Sum explained variance: {np.sum(d_U.explained_variance_ratio_):1.3e} ')
        
        # Compare on the testset
        # data -> encode -> data in reduced dimension -> decode -> reconstructed data
        UF  = d_U.output(d_U.input(fields_sub['U'][:, testIndex]))  + fields_mean['U'][:, testIndex]

        # Plot and save results
        # if os.path.exists('Figures/UF2UL') == False:
        # os.mkdir('Figures/UF2UL')
        ref   = fields['U'][:, testIndex]
        pred  = UF
        # plotPCAComponents(d_U,savefolder = 'Figures/UF2UL/pcaComponent')
        # plotTest(ref,pred,plotSensor=False,savefolder='Figures/UF2UL')
        saveToMatlab_pca(d_U,savename=f'run{runIndex}/pred/TF2TL.mat')

        # -------------------------------------------------------------------------
        # Part 2: Mapping T sparse space to U latent space "TS2UL" 
        # -------------------------------------------------------------------------
        # F_NN2
        # if os.path.exists('Figures/TS2UL') == False:
        #     os.mkdir('Figures/TS2UL') 

        # decode latent space  by PCA
        UL   = d_U.input(fields_sub['U'][:, trainIndex])     # U latent space
        TS   = sensorSampling(fields_sub['T'][:, trainIndex], showTCgrid=False) # T sparse space

        # FNN (MLP) from T latent to U latent space 
        F_NN3 = DNN.DNN(hiddenLayerNodes=[10,10], inputNormalization=True)
        # Early stopping
        F_NN3.earlyStopping = None   # 'None' means empty, not do this, or use 'True' to enable early stopping
        F_NN3.max_epoch     = Train_epoch_2
        # Training 'DNN' to build mapping from 'Tsensor' to PCA latent space ('pca_out')
        F_NN3.train( Xtrain = TS, Ytrain = UL, verbose=True)
        # Plot training history
        # plotTrainHistory(F_NN3.history,savefolder='Figures/TS2UL')

        # Save for matlab post-processing
        # Test data
        TS    = sensorSampling(fields_sub['T'][:, testIndex], showTCgrid=False) # T sparse space
        UL    = F_NN3.output(Xinp = TS) # U latent space by F_NN3
        
        ref   = d_U.input(fields_sub['U'][:, testIndex])
        pred  = UL
        saveToMatlab_dnn(testIndex, sensorPlacements, ref, pred, F_NN3.history, savename=f'run{runIndex}/pred/TL2UL.mat')

        # -------------------------------------------------------------------------
        # Reconstruction T sparse to U full (TS2UF): UF = d_U * F_NN3 * TS 
        # -------------------------------------------------------------------------
        # Reconstruction: 
        # first mapping T sparse sensor data to T latent space
        # prediction
        # if os.path.exists('Figures/TS2UF') == False:
        #     os.mkdir('Figures/TS2UF') 
        UF      = d_U.output(UL) + fields_mean['U'][:, testIndex]
        # Plot and save for matlab post-processing
        ref     = fields['U'][:, testIndex]
        pred    = UF
        # Plot and save results
        # plotTest(ref,pred,plotSensor=True,savefolder='Figures/TS2UF')  
        saveToMatlab_dnn(testIndex, sensorPlacements, ref, pred,F_NN3.history, savename=f'run{runIndex}/pred/TS2UF.mat')