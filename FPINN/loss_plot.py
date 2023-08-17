import numpy as np 
import matplotlib.pyplot as plt
import scipy.io
import scipy as sp

data = np.load(f"pred/res_SquareTest_24_10_tanh_20000_15200_495_Solve_PDE_PlaneJet_NoBuoyancy.npz")

# Provide post processing of the results

# Plot loss history

name_lgd = ['BC','Cont','Mom-x','Mom-y','Energy']

figure = plt.figure(figsize=(8,6))
for i in range(5):
    plt.plot(data['hist'][:, i], label=f'{name_lgd[i]}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
# savefig
plt.savefig('Figures/losses.png')
plt.close()

# Save to matlab file
sp.io.savemat('pred/PINN_results.mat', data)


# pred = pred, x = X, y = Y, hist = hist