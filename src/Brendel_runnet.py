import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.linalg import toeplitz
import time

def runnet(dt, leak, F, Input, C, Nneuron, Ntime, Thresh):
    """
    ネットワークを実行する関数 (学習なし)
    """
    # 初期化
    rO = np.zeros((Nneuron, Ntime)) # filtered spike trains
    O = np.zeros((Nneuron, Ntime))  # spike trains
    V = np.zeros((Nneuron, Ntime))  # membrane potentials
    
    # Pythonでのベクトル計算のために形状を調整 (N, 1)
    # Inputは (Nx, Ntime)
    
    # 時間ループ (t=1 から Ntime-1 まで。MATLABの t=2:Ntime に相当)
    for t in range(1, Ntime):
        # 膜電位の更新
        # V(:,t) = (1-lambda*dt)*V(:,t-1) + dt*F'*Input(:,t-1) + C*O(:,t-1) + noise
        
        # ノイズ生成
        noise = 0.001 * np.random.randn(Nneuron)
        
        # 以前の入力とスパイクを取得
        prev_V = V[:, t-1]
        prev_Input = Input[:, t-1]
        prev_O = O[:, t-1]
        
        # 更新式 (F.T @ prev_Input は行列ベクトル積)
        V[:, t] = (1 - leak * dt) * prev_V + \
                  dt * (F.T @ prev_Input) + \
                  (C @ prev_O) + \
                  noise
        
        # 最大膜電位を持つニューロンを探す
        # MATLAB: [m,k]= max(V(:,t) - Thresh-0.01*randn(Nneuron,1));
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V[:, t] - Thresh - thresh_noise
        
        k = np.argmax(potentials)
        m = potentials[k]
        
        # 発火判定
        if m >= 0:
            O[k, t] = 1
            
        # フィルタリングされたスパイク列の更新
        # rO(:,t) = (1-lambda*dt)*rO(:,t-1) + 1*O(:,t)
        rO[:, t] = (1 - leak * dt) * rO[:, t-1] + 1.0 * O[:, t]
        
    return rO, O, V