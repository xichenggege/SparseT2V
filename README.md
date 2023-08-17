# SparseT2V: Reconstruct flow velocity through sparse temperature measurement
The source code for SCOPE-1 paper X.Wang, Y.Chen et al. Flow reconstruction of single-phase planar jet from sparse temperature measurements (Under review).

## System requirements
1. The main function is written in Python 3.8.0 and tensorlfow 2.8.0. 
2. Code for post-processing is written in Matlab (version R2021b).
2. Training by GPU is enabled by: Tensorflow 2.8.0 + Python 3.8.0 and instructions in (https://towardsdatascience.com/how-to-finally-install-tensorflow-gpu-on-windows-10-63527910f255)

## Installation guide
1. Install Python 3
2. Install generalRegressionFramework (https://github.com/YimengChankth/generalRegressionFramework.git). It is a general framework that contains *PCA*, *MLP* etc., and can call them in a same manner. This is not a public pacakage. To use:
   2.1 Download as zip
   2.2 Open Python cmd, change to your working environment, cd to the root of downloaded package
   2.3 Type 'pip3 install -e .'
4. Download training data (https://kth-my.sharepoint.com/:u:/g/personal/xicheng_ug_kth_se/EVbl54WaIzRPoBcmKBUFGCIB_5UmJnOvNv9ZSE9tAMi1iw?e=W7vaYr) and upzip them under and put all 3 group mat files under:**_.\TrainData_**
5. Optional: install Matlab> vR2021b for post-processing

## Requried libraries
1. numpy
2. scipy
3. os
4. matplotlib
5. Tensorflow
6. time
7. sklearn
8. pickle

## Demo of FDD-1
1. Open **_.FDD-1/main.py_** 
2. Set parameters 
    num_run         = 1       (run 1 times)
    testSplit_ratio = 0.1     (10% testing data)
3. Adjust sensor index by open **_./optSensorPositionT_0999.py_**. It can be obtained by optimal sensor placement in **_./OptSensorPlacement/main.py_**
4. Run the code

All results are stored as matlab file for post-processing
The run time could be between several minutes to several hours depending on the parameters you choose, e.g., the dataset size and the number of iterations for training.

## Demo of FPINN
1. Open **_.FPINN/main.py_** 
2. Define data to be used for training as avaliable information in **_./main.py_** 
```
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

```
and in **_.FPINN/PINN_PlaneJet_NoDimensonal.py_** 
```
    # Loss of known vairable (e.g., BC)
    loss_bc    = self.lambda0*tf.reduce_mean(tf.square(Y_pred_bc[:,0:2] - Y_bc[:,0:2]))  # Only U,V on specific boundary 
    loss_data  = self.lambda0*tf.reduce_mean(tf.square(Y_pred_data[:,3] - Y_data[:,3]))  # Full temperature 
```
3. PINN structure can be revised in **_./PINN_PlaneJet_NoDimensonal.py_** 
4. Setup of Adma and lbfgs optimzes can be revised in **_./train_configs.py_** and **_./lbfgs.py_**
5. Run the code 

## Instructions for use
1. The setup of parameters details for PCA and nerual networks class can be found in **_./generalRegressionFramework/PCA.py_** and **_./generalRegressionFramework/DNN.py**_


## Questions
To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
