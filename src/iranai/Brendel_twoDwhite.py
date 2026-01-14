import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.linalg import toeplitz
import time
from Brendel_learning import learning

def twoDWhite():
    """
    メイン実行関数 (twoDWhite.m に相当)
    """
    Nneuron = 20    # size of the population
    Nx = 2          # dimension of the input
    
    leak = 50       # membrane leak (lambda)
    dt = 0.001      # time step
    
    epsr = 0.001    # learning rate of recurrent connections
    epsf = 0.0001   # learning rate of FF connections
    
    alpha = 0.18    # scaling of FF weights
    beta = 1 / 0.9  # scaling of recurrent weights
    mu = 0.02 / 0.9 # quadratic cost
    
    # Initial connectivity
    # MATLAB: Fi=0.5*randn(Nx,Nneuron);
    Fi = 0.5 * np.random.randn(Nx, Nneuron)
    
    # Normalize FF weights
    # MATLAB: Fi=1*(Fi./(sqrt(ones(Nx,1)*(sum(Fi.^2)))));
    # sum(Fi.^2) -> column sum in MATLAB. axis=0 in NumPy.
    col_norms = np.sqrt(np.sum(Fi**2, axis=0))
    Fi = Fi / col_norms
    
    # Initial Recurrent connectivity
    # MATLAB: Ci=-0.2*(rand(Nneuron,Nneuron))-0.5*eye(Nneuron);
    # rand -> uniform [0,1]
    Ci = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    
    Thresh = 0.5
    
    # Run Learning
    # Note: Added extra returns (Error, Rates) to plotting simpler within python structure
    Fs, Cs, F, C, Decs, ErrorC, Error, MeanPrate, MembraneVar, T_size = learning(
        dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Fi, Ci
    )
    
    # ---------------------------------------------------------
    # Plotting (Learning.m の後半部分)
    # ---------------------------------------------------------
    print("Plotting results...")
    
    times = (2.0 ** np.arange(1, T_size + 1)) * dt
    
    # Figure 1: Performance Stats
    fig1 = plt.figure(figsize=(10, 12))
    
    # Decoding Error
    ax1 = fig1.add_subplot(3, 1, 1)
    ax1.loglog(times, Error, 'k')
    ax1.set_xlabel('time')
    ax1.set_ylabel('Decoding Error')
    ax1.set_title('Evolution of the Decoding Error Through Learning')
    
    # Mean Rate
    ax2 = fig1.add_subplot(3, 1, 2)
    ax2.loglog(times, MeanPrate, 'k')
    ax2.set_xlabel('time')
    ax2.set_ylabel('Mean Rate per neuron')
    ax2.set_title('Evolution of the Mean Population Firing Rate Through Learning')
    
    # Membrane Variance
    ax3 = fig1.add_subplot(3, 1, 3)
    ax3.loglog(times, MembraneVar, 'k')
    ax3.set_xlabel('time')
    ax3.set_ylabel('Voltage Variance per Neuron')
    ax3.set_title('Evolution of the Variance of the Membrane Potential')
    
    plt.tight_layout()
    plt.savefig('results/twoDWhite_performance.png')

    # Figure 2: Weights Analysis
    fig2 = plt.figure(figsize=(12, 14))
    
    # Distance to optimal weights
    # MATLAB: subplot(lines,2,[1 2]) -> spans first row
    ax_w1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax_w1.loglog(times, ErrorC, 'k')
    ax_w1.set_xlabel('time')
    ax_w1.set_ylabel('Distance to optimal weights')
    ax_w1.set_title('Weight Convergence')
    
    # Before Learning: FF Weights
    Fi_start = Fs[0, :, :]
    ax_w2 = plt.subplot2grid((4, 2), (1, 0))
    ax_w2.plot(Fi_start[0, :], Fi_start[1, :], '.k')
    ax_w2.plot(0, 0, '+')
    ax_w2.set_xlim([-1, 1])
    ax_w2.set_ylim([-1, 1])
    ax_w2.set_xlabel('FF Weights Component 1')
    ax_w2.set_ylabel('FF Weights Component 2')
    ax_w2.set_title('Before Learning')
    ax_w2.set_aspect('equal')
    
    # After Learning: FF Weights
    ax_w3 = plt.subplot2grid((4, 2), (1, 1))
    ax_w3.plot(F[0, :], F[1, :], '.k')
    ax_w3.plot(0, 0, '+')
    ax_w3.set_xlim([-1, 1])
    ax_w3.set_ylim([-1, 1])
    ax_w3.set_xlabel('FF Weights Component 1')
    ax_w3.set_ylabel('FF Weights Component 2')
    ax_w3.set_title('After Learning')
    ax_w3.set_aspect('equal')
    
    # Before Learning: Recurrent vs FF^T
    Ci_start = Cs[0, :, :]
    FFT_start = - (Fi_start.T @ Fi_start)
    ax_w4 = plt.subplot2grid((4, 2), (2, 0))
    # MATLAB: plot(Ci, -Fi'*Fi, '.k') -> Flatten arrays for scatter plot equivalent
    ax_w4.plot(Ci_start.flatten(), FFT_start.flatten(), '.k')
    ax_w4.plot(0, 0, '+')
    ax_w4.set_xlim([-1, 1])
    ax_w4.set_ylim([-1, 1])
    ax_w4.set_xlabel('FF^T')
    ax_w4.set_ylabel('Learned Rec Weights')
    ax_w4.set_aspect('equal')
    
    # After Learning: Recurrent vs FF^T
    FFT_end = - (F.T @ F)
    ax_w5 = plt.subplot2grid((4, 2), (2, 1))
    ax_w5.plot(C.flatten(), FFT_end.flatten(), '.k')
    ax_w5.plot(0, 0, '+')
    ax_w5.set_xlim([-1, 1])
    ax_w5.set_ylim([-1, 1])
    ax_w5.set_xlabel('FF^T')
    ax_w5.set_ylabel('Learned Rec Weights')
    ax_w5.set_aspect('equal')
    
    # Before Learning: Decoder vs F^T
    Dec_start = Decs[0, :, :] # (Nx, Nneuron)
    # MATLAB: plot(Dec, Fi, ...)
    ax_w6 = plt.subplot2grid((4, 2), (3, 0))
    ax_w6.plot(Dec_start.flatten(), Fi_start.flatten(), '.k')
    ax_w6.plot(0, 0, '+')
    ax_w6.set_xlim([-1, 1])
    ax_w6.set_ylim([-1, 1])
    ax_w6.set_xlabel('Optimal decoder')
    ax_w6.set_ylabel('F^T')
    ax_w6.set_aspect('equal')
    
    # After Learning: Decoder vs F^T
    Dec_end = Decs[T_size-1, :, :]
    ax_w7 = plt.subplot2grid((4, 2), (3, 1))
    ax_w7.plot(Dec_end.flatten(), F.flatten(), '.k')
    ax_w7.plot(0, 0, '+')
    ax_w7.set_xlim([-1, 1])
    ax_w7.set_ylim([-1, 1])
    ax_w7.set_xlabel('Optimal decoder')
    ax_w7.set_ylabel('F^T')
    ax_w7.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('results/twoDWhite_results.png')

if __name__ == "__main__":
    # コード実行
    twoDWhite()