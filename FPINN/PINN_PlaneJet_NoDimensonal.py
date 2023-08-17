
import numpy as np
import tensorflow as tf
from   tensorflow.keras import models
from   lbfgs import optimizer as lbfgs_op


class PINNs(models.Model):
    def __init__(self, model, optimizer, epochs, **kwargs):
        super(PINNs, self).__init__(**kwargs)
        self.model     = model
        self.optimizer = optimizer
        self.epochs    = epochs
        self.hist      = []
        self.epoch     = 0
        self.sopt      = lbfgs_op(self.trainable_variables)
        # PDE properties
        self.g         = -9.81        # gravitational acceleration [m/s^2]
        self.Pr_t      = 0.85         # Turbulent prantle number

        # Weighting factor of each equation
        self.lambda0 = 50      # for known data
        self.lambda1 = 1       # for Eq 1
        self.lambda2 = 1       # for Eq 2
        self.lambda3 = 1       # for Eq 3
        self.lambda4 = 1       # for Eq 4

    # Main function of PINN, calculate loss of each items     
    @tf.function
    def net_f(self, cp):

        # cp = self.scalex_r(cp)  # recover the collection points since pde is dimensional
        x = cp[:,0] # x    coordinate
        y = cp[:,1] # y    coordinate

        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(x)
            tape.watch(y)
            # Prediction
            inp  = tf.stack([x, y], axis = -1)
            # inp  = self.scalex(inp)        # scale to [0 1]
            pred = self.model(inp)         # prediction by neural network
            # pred = self.scale_r(pred)      # unscale

            # Liquid properties
            rho  = 998.2
            k    = 0.6
            mul  = 0.001003
            Cp   = 4182

            # Non-dimensonal
            U0   = 1.0    # Inlet velocity  [m/s]
            L    = 16e-3  # nozzle diameter [m]
            Re_rev  = 1/(U0*L/(mul/rho))    # 1/Reynolds number
            Pe_rev  = 1/(U0*L/(k/(rho*Cp))) # 1/Peclet number

            U_   = pred[:, 0]
            V_   = pred[:, 1]
            P_   = pred[:, 2]
            T_   = pred[:, 3]
            Ret_rev = pred[:, 4]        # 1/Turbulent Reynolds number
            Pet_rev = Ret_rev/self.Pr_t # 1/Turbulent Peclet number

            # 1st derivative
            U_x    = tape.gradient(U_, x)
            U_y    = tape.gradient(U_, y)
            V_x    = tape.gradient(V_, x)
            V_y    = tape.gradient(V_, y)
            P_x    = tape.gradient(P_, x)
            P_y    = tape.gradient(P_, y)
            T_x    = tape.gradient(T_, x)
            T_y    = tape.gradient(T_, y)

            # 2nd derivative
            U_xx    = tape.gradient(U_x, x)
            U_yy    = tape.gradient(U_y, y)
            V_xx    = tape.gradient(V_x, x)
            V_yy    = tape.gradient(V_y, y)
            T_xx    = tape.gradient(T_x, x)
            T_yy    = tape.gradient(T_y, y)
 
            # Build up the loss function of    
            # Continuty equation
            f1 = U_x + V_y
            # Momentum equations X Y         
            f2 = U_ * U_x + V_ * U_y + P_x - (Re_rev + Ret_rev)*(U_xx + U_yy)
            f3 = U_ * V_x + V_ * V_y + P_y - (Re_rev + Ret_rev)*(V_xx + V_yy) 
            # Energy equation
            f4 = U_ * T_x + V_ * T_y - (Pe_rev + Pet_rev) * (T_xx + T_yy) 

        return f1, f2, f3, f4
  
    @tf.function
    def train_step(self, bc, fdata, cp):
        
        # bc and cp are already scaled variables
        X_bc     = bc[:, :2]  # first 2 columns are X,Y mesh points
        Y_bc     = bc[:, 2:]  # last  5 columnsUse are U,V,P,T,Ret_rev
        X_data   = fdata[:, :2]  # first 2 columns are X,Y mesh points
        Y_data   = fdata[:, 2:]  # last  5 columnsUse are U,V,P,T,Ret_rev
        with tf.GradientTape() as tape:
            
            # Prediction by NN
            Y_pred_bc    = self.model(X_bc)        # bc
            Y_pred_data  = self.model(X_data)      # data
            f1, f2, f3, f4    = self.net_f(cp)     # cp
            # Loss of known vairable (e.g., BC)
            loss_bc    = self.lambda0*tf.reduce_mean(tf.square(Y_pred_bc[:,0:2] - Y_bc[:,0:2]))  # Only U,V on specific boundary 
            loss_data  = self.lambda0*tf.reduce_mean(tf.square(Y_pred_data[:,3] - Y_data[:,3]))  # Full temperature 
            # Loss of pde
            loss_f1    = self.lambda1*tf.reduce_mean(tf.square(f1)) 
            loss_f2    = self.lambda2*tf.reduce_mean(tf.square(f2)) 
            loss_f3    = self.lambda3*tf.reduce_mean(tf.square(f3)) 
            loss_f4    = self.lambda4*tf.reduce_mean(tf.square(f4))  
            loss_f     = loss_f1 + loss_f2 + loss_f3 + loss_f4
            loss       = loss_bc + loss_data + loss_f
            
        trainable_vars = self.trainable_variables
        grads          = tape.gradient(loss, trainable_vars)
            
        tf.print('loss_all:', loss, 'loss_bc:', loss_bc + loss_data, 'loss_f:', loss_f)
        return loss, grads, tf.stack([loss_bc, loss_data, loss_f1, loss_f2, loss_f3, loss_f4])
    
    # training
    def fit(self, bc, fdata, cp):

        bc    = tf.convert_to_tensor(bc, tf.float32)
        fdata = tf.convert_to_tensor(fdata, tf.float32)
        cp    = tf.convert_to_tensor(cp, tf.float32)
        

        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, fdata, cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(hist.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        ############################################################
        # Trainning 
        ############################################################
        # adam train first N epochs
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, fdata, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(hist.numpy())

        # then call L-BFGS-B 
        self.sopt.minimize(func)
            
        return np.array(self.hist)
    
    def predict(self, cp):
        cp = tf.convert_to_tensor(cp, tf.float32)
        # cp = self.scalex(cp)
        u_p = self.model(cp)
        # u_p = self.scale_r(u_p)
        return u_p.numpy()