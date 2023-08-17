import numpy as np
import scipy.io
import os
from tensorflow.keras import models, layers, optimizers, activations
from PINN_PlaneJet_NoDimensonal import PINNs
from matplotlib import pyplot as plt
from time import time
from train_configs import PINN_config
from error import l2norm_err

"""
Train PINN for Single phase plane jet 2D
"""


#################
# DATA loading
#################
def getData():

    dataLoc     = os.getcwd()
    case_name   = '20230627_PlaneJet2D_Benchmark_nonDimensonal.mat'
    filePath    = f'{dataLoc[0:len(dataLoc)]}TrainData\\{case_name}'

    mat         = scipy.io.loadmat(filePath)
    Nsamples    = mat['T_'].shape[1]
    print(f'Number of samples: {Nsamples}')
    fields = {}

    # Duplitcate x y t and save T U V P

    for var in ['T_','U_','V_','P_','Ret_rev']:
        print(f'Variable: {var}, shape: {mat[var][0,0].shape}')
        fields[var] = np.zeros(( mat[var][0,0].flatten().shape[0] , Nsamples)) 
        # Ensure that the number of samples is the same for every field
        assert Nsamples == mat[var].shape[1]
        for n in range(Nsamples):
           fields[var][:, n] = mat[var][0, n].flatten()
    
    for var in ['xmesh_','ymesh_']:
        print(f'Variable: {var}, shape: {mat[var][0,0].shape}')
        fields[var] = np.zeros((mat[var].flatten().size , Nsamples)) 
        for n in range(Nsamples):
            fields[var][:, n] = mat[var].flatten()


    # Concatenate every variable into single column
    X = np.stack((fields['xmesh_'].T.flatten(), fields['ymesh_'].T.flatten()), axis=1)
    Y = np.stack((fields['U_'].T.flatten(), fields['V_'].T.flatten(), fields['P_'].T.flatten(),fields['T_'].T.flatten(), fields['Ret_rev'].T.flatten()), axis=1)
    print(f'\tShape of X: {X.shape}')
    print(f'\tShape of Y: {Y.shape}')

    return X, Y

def plot_result(X, Ynp, Nx, Ny, figType,saveName):

    # Color limit
    vminU = np.min(Ynp[:,0])
    vmaxU = np.max(Ynp[:,0])

    vminV = np.min(Ynp[:,1])
    vmaxV = np.max(Ynp[:,1])

    vminP = np.min(Ynp[:,2])
    vmaxP = np.max(Ynp[:,2])

    vminT = np.min(Ynp[:,3])
    vmaxT = np.max(Ynp[:,3])

    vminRet = np.min(Ynp[:,4])
    vmaxRet = np.max(Ynp[:,4])

    fig,axs = plt.subplots(3,2,figsize=(15,10))
    if (figType == 'contourf'):
        xmesh = X[:,0].reshape(Ny,Nx)
        ymesh = X[:,1].reshape(Ny,Nx)
        plotU = axs[0,0].contourf(xmesh,ymesh,Ynp[:,0].reshape(Ny,Nx),cmap='jet',vmin=vminU,vmax=vmaxU)
        plotV = axs[0,1].contourf(xmesh,ymesh,Ynp[:,1].reshape(Ny,Nx),cmap='jet',vmin=vminV,vmax=vmaxV)
        plotP = axs[1,0].contourf(xmesh,ymesh,Ynp[:,2].reshape(Ny,Nx),cmap='jet',vmin=vminP,vmax=vmaxP)
        plotT = axs[1,1].contourf(xmesh,ymesh,Ynp[:,3].reshape(Ny,Nx),cmap='jet',vmin=vminT,vmax=vmaxT)
        plotRet = axs[2,0].contourf(xmesh,ymesh,Ynp[:,4].reshape(Ny,Nx),cmap='jet',vmin=vminRet,vmax=vmaxRet)
  
    elif(figType == 'scatter'):
        plotU = axs[0,0].scatter(x=X[:,0],y=X[:,1],c=Ynp[:,0],cmap='jet',vmin=vminU,vmax=vmaxU)
        plotV = axs[0,1].scatter(x=X[:,0],y=X[:,1],c=Ynp[:,1],cmap='jet',vmin=vminV,vmax=vmaxV)
        plotP = axs[1,0].scatter(x=X[:,0],y=X[:,1],c=Ynp[:,2],cmap='jet',vmin=vminP,vmax=vmaxP)
        plotT = axs[1,1].scatter(x=X[:,0],y=X[:,1],c=Ynp[:,3],cmap='jet',vmin=vminT,vmax=vmaxT)
        plotRet = axs[2,0].scatter(x=X[:,0],y=X[:,1],c=Ynp[:,4],cmap='jet',vmin=vminRet,vmax=vmaxRet)

    axs[0,0].set_title('U')
    plt.colorbar(plotU, ax=axs[0,0])
    axs[0,1].set_title('V')
    plt.colorbar(plotV,ax=axs[0,1])
    axs[1,0].set_title('P')
    plt.colorbar(plotP,ax=axs[1,0])
    axs[1,1].set_title('T')
    plt.colorbar(plotT,ax=axs[1,1])
    axs[2,0].set_title('1/Ret')
    plt.colorbar(plotRet,ax=axs[2,0])
    plt.savefig(saveName,dpi=300)
    print(f'Plotted to: {saveName}')

    pass

if __name__ == "__main__":

    # Load total tranining data from CFD
    X, Y = getData()

    Nx = 160 
    Ny = 95

    # Preview your loaded data
    Ynp = Y
    plot_result(X, Ynp,Nx,Ny,'contourf','Figures/Field_input_values.png')

    ###############################################################################################
    # Define your problem, i.e. inverse or forward
    # 1. Forward problem -> known pde, boundary info to solve pde (train for inernal point)
    # 2. inverse problem -> known some full domain data, part of pde, to optimize other results
    ###############################################################################################
    # Select collection points (cp) 
    cp   = X
    n_cp = len(cp)

    # Collect known info as boundary condition
    idx_bc = np.zeros((Ny,Nx), dtype = bool) # note the order of Nx and Ny
    idx_bc[:, 0] = True  # left
    # Downstream PIV window
    idx_bc[30:50, -50:-30] = True  # left to right

    # Assume known every variables on boundary
    idx_bc = idx_bc.flatten()
    X_bc   = X[idx_bc,:]
    Y_bc   = Y[idx_bc,:]

    # data -> full temperature 
    idx_data = np.ones((Ny,Nx), dtype = bool) # note the order of Nx and Ny
    idx_data = idx_data.flatten()
    X_data   = X[idx_data,:]
    Y_data   = Y[idx_data,:]

    plot_result(X_bc, Y_bc,Nx,Ny,'scatter','Figures/boundary_vals.png')

    # each column represents a variable
    bc    = np.concatenate((X_bc, Y_bc), axis=1)
    fdata = np.concatenate((X_data, Y_data), axis=1)

    ###############################################
    # Setup training Parameters
    ###############################################
    act     = PINN_config.act
    nn      = PINN_config.n_neural
    nl      = PINN_config.n_layer
    n_adam  = PINN_config.n_adam
    cp_step = PINN_config.cp_step
    bc_step = PINN_config.bc_step

    # record the test name
    n_bc = len(bc)
    test_name = f'Test_{nn}_{nl}_{act}_{n_adam}_{n_cp}_{n_bc}_Solve_PDE_PlaneJet_NoBuoyancy'
    
    ####################################################################
    # Compiling Model
    ####################################################################
    num_input  = 2
    num_output = 5

    inp = layers.Input(shape = (num_input,))
    hl  = inp
    for i in range(nl):
        hl = layers.Dense(nn, activation = act)(hl)
    out = layers.Dense(num_output)(hl)
    model = models.Model(inp, out)

    print(model.summary())
    lr = 1e-3
    opt = optimizers.Adam(lr)
    pinn = PINNs(model, opt, n_adam)

    #########################
    # Training
    #########################
    print(f"INFO: Start training case_{test_name}")
    st_time   = time()
    hist      = pinn.fit(bc, fdata, cp)

    en_time   = time()
    comp_time = en_time - st_time
    # %%
    #########################
    # Prediction
    #########################
    cpp  = X
    pred = pinn.predict(cpp)
    Ynp  = pred
    plot_result(X, Ynp, Nx,Ny,'contourf','Figures/' + test_name)

    #################
    # Save prediction and Model
    #################
    np.savez_compressed('pred/res_Square' + test_name, pred = pred, x = X, y = Y, hist = hist, ct = comp_time)
    model.save('models/model_Square' + test_name + '.h5')
    print("INFO: Prediction and model have been saved!")