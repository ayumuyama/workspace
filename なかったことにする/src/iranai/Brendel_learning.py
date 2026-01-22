import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.linalg import toeplitz
import time
from iranai.Brendel_runnet import runnet

def learning(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, F, C):
    """
    再帰結合とフィードフォワード結合の学習を行う関数
    """
    print("Starting Learning...")
    
    # 学習パラメータ
    Nit = 14000      # iteration count
    Ntime = 1000     # size of input sequence
    TotTime = Nit * Ntime # total time
    
    # 記録用配列のサイズ計算
    # MATLAB: T=floor(log(TotTime)/log(2));
    T_size = int(np.floor(np.log2(TotTime)))
    
    Cs = np.zeros((T_size, Nneuron, Nneuron))
    Fs = np.zeros((T_size, Nx, Nneuron))
    
    # 状態変数の初期化
    V = np.zeros(Nneuron)
    O = 0 # fire flag
    k = 0 # firing neuron index
    rO = np.zeros(Nneuron)
    x = np.zeros(Nx) # filtered input
    
    # 単位行列
    Id = np.eye(Nneuron)
    
    # 入力生成用パラメータ
    A = 2000
    sigma = 30
    
    # ガウシアンカーネルの作成
    # MATLAB: w=(1/(sigma*sqrt(2*pi)))* exp(-(([1:1000]-500).^2)/(2*sigma.^2));
    t_kern = np.arange(1, 1001)
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_kern - 500)**2) / (2 * sigma**2))
    w = w / np.sum(w)
    
    j = 0 # Python index (0-based) for exponential logging
    l = 1 # percent counter
    
    print('0 percent of the learning completed')
    
    Input = np.zeros((Nx, Ntime))
    
    # メイン学習ループ
    # MATLAB: for i=2:TotTime
    # Python: range(1, TotTime) とし、ループ内変数を調整
    for i in range(1, TotTime):
        
        # 進捗表示
        if (i / TotTime) > (l / 100.0):
            print(f'{l} percent of the learning completed')
            l += 1
            
        # 指数関数的なタイミングで重みを記録
        # MATLAB: mod(i, 2^j) == 0. Pythonのjは0始まりなので調整が必要
        # MATLABのj=1は 2^1=2。Pythonでここに入るのは i=2の時。
        # ここではMATLABのロジック (2, 4, 8...) に合わせる
        current_pow2 = 2**(j + 1)
        if i % current_pow2 == 0:
            if j < T_size:
                Cs[j, :, :] = C.copy()
                Fs[j, :, :] = F.copy()
                j += 1
        
        # 新しい入力シーケンスの生成 (Ntimeごとに更新)
        # MATLAB: mod(i-2, Ntime) == 0 -> i=2, 1002, ...
        # Python: i=1スタートなので (i-1) % Ntime == 0 ? あるいは単に i % Ntime == 1 ?
        # MATLABのロジック: i=2のとき生成、以降+Ntimeごと。
        # Python loop i (1...): i=1 (t=0相当) の次は i=2 (t=1相当)。
        # MATLABとの整合性を取るため、(i - 1) % Ntime == 0 のタイミングで生成する
        if (i - 1) % Ntime == 0:
            # (Ntime, Nx) の正規乱数
            raw_input = np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), Ntime)
            Input = raw_input.T # (Nx, Ntime)
            
            # スムージング
            for d in range(Nx):
                # MATLAB: conv(..., 'same')
                Input[d, :] = A * convolve(Input[d, :], w, mode='same')
        
        # 現在のタイムステップのインデックス (0 to Ntime-1)
        # MATLAB: mod(i, Ntime)+1. 
        # Python: i % Ntime 
        t_idx = i % Ntime 
        
        curr_Input = Input[:, t_idx]
        
        # 膜電位更新
        # V = (1-lambda*dt)*V + dt*F'*Input + O*C(:,k) + noise
        noise = 0.001 * np.random.randn(Nneuron)
        
        # C[:, k] は前のステップで発火したニューロンkからの入力。
        # Oが1（発火あり）の場合のみ加算。
        recurrent_input = np.zeros(Nneuron)
        if O == 1:
            recurrent_input = C[:, k]
            
        V = (1 - leak * dt) * V + dt * (F.T @ curr_Input) + recurrent_input + noise
        
        # フィルタ入力更新
        x = (1 - leak * dt) * x + dt * curr_Input
        
        # 最大電位探索
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        
        k_curr = np.argmax(potentials)
        m = potentials[k_curr]
        
        if m >= 0:
            O = 1
            k = k_curr # update firing neuron index
            
            # 重み更新
            # F(:,k) = F(:,k) + epsf*(alpha*x - F(:,k))
            F[:, k] += epsf * (alpha * x - F[:, k])
            
            # C(:,k) = C(:,k) - epsr * (beta*(V + mu*rO) + C(:,k) + mu*Id(:,k))
            C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
            
            # フィルタ済みスパイク列更新
            rO[k] += 1
        else:
            O = 0
            # kは更新しない（前回値を保持、ただしO=0なのでC[:,k]は使われない）
            
        # フィルタ済みスパイク列の減衰
        rO = (1 - leak * dt) * rO
        
    print('Learning completed')
    
    # ---------------------------------------------------------
    # Computing Optimal Decoders
    # ---------------------------------------------------------
    print('Computing optimal decoders')
    TimeL = 50000
    xL = np.zeros((Nx, TimeL))
    Decs = np.zeros((T_size, Nx, Nneuron))
    
    # Generate InputL
    raw_inputL = 0.3 * A * np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeL)
    InputL = raw_inputL.T
    
    for d in range(Nx):
        InputL[d, :] = convolve(InputL[d, :], w, mode='same')
        
    # Compute target output xL
    for t in range(1, TimeL):
        xL[:, t] = (1 - leak * dt) * xL[:, t-1] + dt * InputL[:, t-1]
        
    # 各保存済み重みについてデコーダを計算
    for i_idx in range(T_size):
        curr_F = Fs[i_idx, :, :]
        curr_C = Cs[i_idx, :, :]
        
        rOL, _, _ = runnet(dt, leak, curr_F, InputL, curr_C, Nneuron, TimeL, Thresh)
        
        # MATLAB: Dec = (rOL'\xL')'
        # これは方程式 rOL.T * Dec.T = xL.T を解くことと同じ (Ax = B)
        # Python: lstsq(A, B) -> A=rOL.T, B=xL.T
        
        # rOLは (Nneuron, TimeL) -> Transpose -> (TimeL, Nneuron)
        # xLは (Nx, TimeL) -> Transpose -> (TimeL, Nx)
        
        solution, residuals, rank, s = np.linalg.lstsq(rOL.T, xL.T, rcond=None)
        Dec = solution.T # (Nx, Nneuron)
        Decs[i_idx, :, :] = Dec

    # ---------------------------------------------------------
    # Computing Decoding Error, rates through Learning
    # ---------------------------------------------------------
    print('Computing decoding errors and rates over learning')
    TimeT = 10000
    MeanPrate = np.zeros(T_size)
    Error = np.zeros(T_size)
    MembraneVar = np.zeros(T_size)
    xT = np.zeros((Nx, TimeT))
    
    Trials = 10
    
    for r in range(Trials):
        # Test Input Generation
        raw_inputT = A * np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeT)
        InputT = raw_inputT.T
        for d in range(Nx):
            InputT[d, :] = convolve(InputT[d, :], w, mode='same')
            
        # Target Output xT
        xT.fill(0) # reset
        for t in range(1, TimeT):
            xT[:, t] = (1 - leak * dt) * xT[:, t-1] + dt * InputT[:, t-1]
            
        xT_var_sum = np.sum(np.var(xT, axis=1)) # sum(var(xT, 0, 2))
            
        for i_idx in range(T_size):
            curr_F = Fs[i_idx, :, :]
            curr_C = Cs[i_idx, :, :]
            curr_Dec = Decs[i_idx, :, :]
            
            rOT, OT, VT = runnet(dt, leak, curr_F, InputT, curr_C, Nneuron, TimeT, Thresh)
            
            # Decode
            xestc = curr_Dec @ rOT
            
            # Error Calculation
            # MATLAB: sum(var(xT-xestc,0,2))
            diff_var_sum = np.sum(np.var(xT - xestc, axis=1))
            Error[i_idx] += diff_var_sum / (xT_var_sum * Trials)
            
            # Mean Rate
            # MATLAB: sum(sum(OT))/(TimeT*dt*Nneuron*Trials)
            MeanPrate[i_idx] += np.sum(OT) / (TimeT * dt * Nneuron * Trials)
            
            # Membrane Variance
            # MATLAB: sum(var(VT,0,2))/(Nneuron*Trials)
            MembraneVar[i_idx] += np.sum(np.var(VT, axis=1)) / (Nneuron * Trials)

    # ---------------------------------------------------------
    # Computing distance to Optimal weights
    # ---------------------------------------------------------
    ErrorC = np.zeros(T_size)
    
    for i_idx in range(T_size):
        CurrF = Fs[i_idx, :, :]
        CurrC = Cs[i_idx, :, :]
        
        Copt = - (CurrF.T @ CurrF) # - F^T * F
        
        # MATLAB: trace(CurrC'*Copt) -> np.trace(CurrC.T @ Copt) -> sum(sum(CurrC * Copt))
        numerator = np.sum(CurrC * Copt) 
        denominator = np.sum(Copt**2)
        
        optscale = numerator / denominator
        Cnorm = np.sum(CurrC**2)
        
        ErrorC[i_idx] = np.sum((CurrC - optscale * Copt)**2) / Cnorm
        
    return Fs, Cs, F, C, Decs, ErrorC, Error, MeanPrate, MembraneVar, T_size