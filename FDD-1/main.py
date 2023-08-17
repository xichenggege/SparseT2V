# Note PCA is performed with sklearn package
# Neural network is performed with tensorflow
# Author: Yi Meng Chan & Xicheng Wang

# generalRegressionFramework is a private github repo found at: YimengChankth/generalRegressionFramework

import generalRegressionFramework.PCA as PCA
import generalRegressionFramework.DNN as DNN

import numpy as np
import numpy.matlib
import scipy.io
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp


# Enter in the shape of the data here
Naxial  = 160
Nradial = 95
Train_epoch_1 = 10000
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
    folder_name = 'FDD-1'
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
            axs[0].scatter(xmesh.flatten()[sensorPlacements], ymesh.flatten()[sensorPlacements], c='r', s=5)

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

    # Subtract mean
    fields_sub   = {}
    fields_mean  = {}
    for var in ['T','U']:
        fields_sub[var], fields_mean[var] = submean(fields[var])        

    Nsamples              = fields_sub['T'].shape[1]
    # Load sensor index by optimal sensor placement
    sensorPlacements      = np.load('optSensorPositionT_0999.npy')
    # index of training and test dataset
    trainIndex, testIndex = trainTestIndex(Nsamples,testSplit_ratio=0.1)   
    # -------------------------------------------------------------------------
    # Run n times for each setup
    # -------------------------------------------------------------------------
    num_run = 1
    for runIndex in range(num_run):
        # Create folder to save each run results
        if os.path.exists(f'run{runIndex}') == False:
            os.mkdir(f'run{runIndex}')
        if os.path.exists(f'run{runIndex}/pred') == False:
            os.mkdir(f'run{runIndex}/pred')
        if os.path.exists(f'run{runIndex}/Figures') == False:
            os.mkdir(f'run{runIndex}/Figures')

        # -------------------------------------------------------------------------
        # Part 1: Decode T by PCA(POD) (dimensional reduction)
        # -------------------------------------------------------------------------
        # PCA with explained variance ratio = 0.999 
        # Set up PCA and train with explained variance ratio 0.999. normalize == True tends to `flatten` the importance of individual meshes. 
        # Decoder d_T
        d_T = PCA.PCA(max_components=100, min_components=1, explained_variance_ratio_threshold=0.999, normalize=False)
        d_T.train(train = fields_sub['T'][:, trainIndex])       # train PCA
        print(f'PCA # of components: {d_T.n_components_} .Sum explained variance: {np.sum(d_T.explained_variance_ratio_):1.3e} ')
        
        # data -> encode -> data in reduced dimension -> decode -> reconstructed data
        # Error estimation
        TF = d_T.output(d_T.input(fields_sub['T'][:, testIndex])) + fields_mean['T'][:, testIndex]
        # Plot and save results
        if os.path.exists(f'run{runIndex}/Figures/TF2TL') == False:
            os.mkdir(f'run{runIndex}/Figures/TF2TL')
        plotPCAComponents(d_T,savefolder = f'run{runIndex}/Figures/TF2TL/pcaComponent')
        # Plot and save for matlab post-processing
        ref   = fields['T'][:, testIndex]
        pred  = TF
        plotTest(ref,pred,plotSensor=False,savefolder=f'run{runIndex}/Figures/TF2TL')
        # Save for matlab post-processing
        saveToMatlab_pca(d_T,savename=f'run{runIndex}/pred/TF2TL.mat')

        # -------------------------------------------------------------------------
        # Part 2: Mapping T sparse (measurement) space to T latent space "TS2TL"
        # -------------------------------------------------------------------------
        # F_NN1   
        if os.path.exists(f'run{runIndex}/Figures/TS2TL') == False:
            os.mkdir(f'run{runIndex}/Figures/TS2TL')
        # sparse sensors
        TS = sensorSampling(fields_sub['T'][:, trainIndex], showTCgrid=True, savefolder= f'run{runIndex}/Figures/TS2TL')
        print(f'Shape of training data: {TS.shape}')
        # set up neural network (MLP) but now using 15 dimensional inputs from PCA
        # Latent space output by PCA
        TL = d_T.input(fields_sub['T'][:, trainIndex])
        # FNN (MLP) from sparse to latent space of 'T'
        F_NN1 = DNN.DNN(hiddenLayerNodes=[10,10], inputNormalization=True)
        # Early stopping
        F_NN1.earlyStopping = None   # 'None' means empty, not do this, or use 'True' to enable early stopping
        F_NN1.max_epoch     = Train_epoch_1
        # Training 'DNN' to build mapping from 'Tsensor' to PCA latent space ('pca_out')
        F_NN1.train( Xtrain = TS, Ytrain = TL, verbose=True)
        # Plot training history
        plotTrainHistory(F_NN1.history,savefolder=f'run{runIndex}/Figures/TS2TL')
        
        # Save for matlab post-processing
        ref   = d_T.input(fields_sub['T'][:, testIndex])
        pred  = F_NN1.output(Xinp = sensorSampling(fields_sub['T'][:, testIndex], showTCgrid=False))
        saveToMatlab_dnn(testIndex, sensorPlacements, ref, pred, F_NN1.history,savename=f'run{runIndex}/pred/TS2TL.mat')

        # -------------------------------------------------------------------------
        # Reconstruction T sparse to T full: TF = d_T * F_NN1 * TS
        # -------------------------------------------------------------------------
        # Reconstruction: 
        # first mapping sparse sensor data to latent space
        # prediction
        if os.path.exists(f'run{runIndex}/Figures/TS2TF') == False:
            os.mkdir(f'run{runIndex}/Figures/TS2TF')
        TS   = sensorSampling(fields_sub['T'][:, testIndex], showTCgrid=False)
        TL   = F_NN1.output(Xinp = TS)
        # Second, decode latent space and obtain complete space
        TF   = d_T.output(TL) + fields_mean['T'][:, testIndex]
        # Plot and save for matlab post-processing
        ref   = fields['T'][:, testIndex]
        pred  = TF
        # plotTest(ref,pred ,plotSensor=True,savefolder='Figures/TS2TF')
        saveToMatlab_dnn(testIndex, sensorPlacements, ref, pred, F_NN1.history, savename=f'run{runIndex}/pred/TS2TF.mat')

        # -------------------------------------------------------------------------
        # Part 3: U full space to U latent space by PCA
        # -------------------------------------------------------------------------
        # PCA with explained variance ratio = 0.999 
        # Set up PCA and train with explained variance ratio 0.999. normalize == True tends to `flatten` the importance of individual meshes. 

        d_U = PCA.PCA(max_components=100, min_components=1, explained_variance_ratio_threshold=0.999, normalize=False)
        d_U.train(train = fields_sub['U'][:, trainIndex])       # train PCA
        print(f'PCA # of components: {d_U.n_components_} .Sum explained variance: {np.sum(d_U.explained_variance_ratio_):1.3e} ')
        
        # Compare on the testset
        # data -> encode -> data in reduced dimension -> decode -> reconstructed data
        UF  = d_U.output(d_U.input(fields_sub['U'][:, testIndex])) + fields_mean['U'][:, testIndex]

        # Plot and save results
        if os.path.exists(f'run{runIndex}/Figures/UF2UL') == False:
           os.mkdir(f'run{runIndex}/Figures/UF2UL')
        ref   = fields['U'][:, testIndex]
        pred  = UF
        plotPCAComponents(d_U,savefolder = f'run{runIndex}/Figures/UF2UL/pcaComponent')
        plotTest(ref,pred,plotSensor=False,savefolder=f'run{runIndex}/Figures/UF2UL')
        saveToMatlab_pca(d_U,savename=f'run{runIndex}/pred/UF2UL.mat')

        # -------------------------------------------------------------------------
        # Part 4: Mapping T latent space to U latent space "TL2UL" 
        # -------------------------------------------------------------------------
        # F_NN2
        if os.path.exists(f'run{runIndex}/Figures/TL2UL') == False:
            os.mkdir(f'run{runIndex}/Figures/TL2UL') 

        # decode latent space  by PCA
        UL   = d_U.input(fields_sub['U'][:, trainIndex])     # U latent space
        TS   = sensorSampling(fields_sub['T'][:, trainIndex], showTCgrid=False) # T sparse space
        TL   = F_NN1.output(Xinp = TS) # T latent space by F_NN1
        # FNN (MLP) from T latent to U latent space 
        F_NN2 = DNN.DNN(hiddenLayerNodes=[10,10], inputNormalization=True)
        # Early stopping
        F_NN2.earlyStopping = None   # 'None' means empty, not do this, or use 'True' to enable early stopping
        F_NN2.max_epoch     = Train_epoch_2
        # Training 'DNN' to build mapping from 'Tsensor' to PCA latent space ('pca_out')
        F_NN2.train( Xtrain = TL, Ytrain = UL, verbose=True)
        # Plot training history
        plotTrainHistory(F_NN2.history,savefolder=f'run{runIndex}/Figures/TL2UL')

        # Save for matlab post-processing
        TS    = sensorSampling(fields_sub['T'][:, testIndex], showTCgrid=False) # T sparse space
        TL    = F_NN1.output(Xinp = TS) # T latent space by F_NN1
        UL    = F_NN2.output(Xinp = TL)

        ref   = d_U.input(fields_sub['U'][:, testIndex])
        pred  = F_NN2.output(Xinp = TL)
        saveToMatlab_dnn(testIndex, sensorPlacements, ref, pred, F_NN2.history, savename=f'run{runIndex}/pred/TL2UL.mat')

        # -------------------------------------------------------------------------
        # Reconstruction T sparse to U full (TS2UF): UF = d_U * F_NN2 * F_NN1 * TS 
        # -------------------------------------------------------------------------
        # Reconstruction: 
        # first mapping T sparse sensor data to T latent space
        # prediction
        if os.path.exists(f'run{runIndex}/Figures/TS2UF') == False:
            os.mkdir(f'run{runIndex}/Figures/TS2UF') 
        UF      = d_U.output(UL) + fields_mean['U'][:, testIndex]
        # Plot and save for matlab post-processing
        ref     = fields['U'][:, testIndex]
        pred    = UF
        # Plot and save results
        plotTest(ref,pred,plotSensor=True,savefolder=f'run{runIndex}/Figures/TS2UF')  
        saveToMatlab_dnn(testIndex, sensorPlacements, ref, pred,F_NN2.history, savename=f'run{runIndex}/pred/TS2UF.mat')