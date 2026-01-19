import numpy as np
from scipy.signal import convolve
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# === 1. 重み初期化 (全ファイル共通) ===
def init_weights(Nx, Nneuron):
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / (np.sqrt(np.sum(F**2, axis=0)) + 1e-8)
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    return F, C

# === 2. MNISTデータ読み込み (MNIST系共通) ===
def load_mnist_data():
    print("Loading MNIST data...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    except:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        
    X = mnist.data
    y = mnist.target.astype(int)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

# === 3. 構造学習: 信号データ用 (Ntimeを引数化) ===
def train_snn_structure_signal(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, 
                               Nit=50000, Ntime=1000): # <--- Ntimeを追加
    """
    Nit: 学習の反復回数 (入力パターンの切り替え回数)
    Ntime: 1つの入力パターンを提示するステップ数
    """
    print(f"Phase 1: Pre-training SNN structure (Signal mode, Nit={Nit}, Ntime={Ntime})...")
    
    TotTime = Nit * Ntime
    F, C = init_weights(Nx, Nneuron)
    
    V = np.zeros(Nneuron)
    O = 0
    k = 0
    rO = np.zeros(Nneuron)
    x = np.zeros(Nx)
    Id = np.eye(Nneuron)
    
    # 入力生成用パラメータ (平滑化カーネル)
    A = 2000
    sigma = 30
    # カーネル長はNtimeに依存させず固定でも良いが、Ntimeより短い必要がある
    # ここでは従来通り1000ステップ分のカーネルを使用
    t_kern = np.arange(1, 1001) 
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_kern - 500)**2) / (2 * sigma**2))
    w = w / np.sum(w)
    
    Input = np.zeros((Nx, Ntime))
    l = 1

    for i in range(1, TotTime):
        # 進捗表示
        if (i / TotTime) > (l / 100.0):
            print(f'\r  Phase 1 Progress: {l}%', end='')
            l += 1
        
        # Ntimeごとに新しい入力を生成
        if (i - 1) % Ntime == 0:
            # Nx次元のランダム入力を生成
            raw_input = np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), Ntime)
            Input = raw_input.T
            # 時間方向に平滑化
            for d in range(Nx):
                Input[d, :] = A * convolve(Input[d, :], w, mode='same')
        
        curr_Input = Input[:, i % Ntime]
        
        noise = 0.001 * np.random.randn(Nneuron)
        recurrent_input = np.zeros(Nneuron)
        if O == 1:
            recurrent_input = C[:, k]
            
        V = (1 - leak * dt) * V + dt * (F.T @ curr_Input) + recurrent_input + noise
        x = (1 - leak * dt) * x + dt * curr_Input
        
        V = np.clip(V, -10, 10) # クリップ処理

        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        k_curr = np.argmax(potentials)
        
        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            
            F[:, k] += epsf * (alpha * x - F[:, k])
            C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
            
            rO[k] += 1
        else:
            O = 0
            
        rO = (1 - leak * dt) * rO
        
    print("\nPhase 1 Completed.")
    return F, C

# === 4. 構造学習: MNIST画像用 (Durationを引数化) ===
def train_snn_structure_mnist(X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain, 
                              Nit=50000, Duration=30): # <--- Durationを追加
    """
    Nit: 画像提示枚数
    Duration: 1枚あたりの提示ステップ数
    """
    print(f"Phase 1: Pre-training SNN structure (MNIST mode, Nit={Nit}, Duration={Duration})...")
    
    F, C = init_weights(Nx, Nneuron)
    
    V = np.zeros(Nneuron)
    O = 0
    k = 0
    rO = np.zeros(Nneuron)
    x = np.zeros(Nx)
    Id = np.eye(Nneuron)
    
    total_spikes = 0 

    for i in range(Nit):
        if i % 100 == 0:
            print(f'\r  Phase 1 Iter: {i}/{Nit} (Spikes: {total_spikes})', end='')
            
        img_idx = np.random.randint(0, len(X_train))
        current_image = X_train[img_idx] * Gain 
        
        for t in range(Duration):
            noise = 0.02 * np.random.randn(Nneuron)
            recurrent_input = np.zeros(Nneuron)
            if O == 1:
                recurrent_input = C[:, k]
                
            V = (1 - leak * dt) * V + dt * (F.T @ current_image) + recurrent_input + noise
            x = (1 - leak * dt) * x + dt * current_image
            V = np.clip(V, -10, 10)

            thresh_noise = 0.01 * np.random.randn(Nneuron)
            potentials = V - Thresh - thresh_noise
            k_curr = np.argmax(potentials)
            
            if potentials[k_curr] >= 0:
                O = 1
                k = k_curr
                total_spikes += 1
                
                F[:, k] += epsf * (alpha * x - F[:, k])
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
                
                rO[k] += 1
            else:
                O = 0
                
            rO = (1 - leak * dt) * rO
            
    print(f"\nPhase 1 Completed. Total spikes: {total_spikes}")
    return F, C

# === 5. 読み出し層学習　MNIST用===
def train_readout_mnist(F, C, X_data, y_data, Nneuron, Nx, dt, leak, Thresh, Gain, label_suffix="", NumSamples=5000, Duration=30, lr_readout=0.02):
    Nclasses = 10
    print(f"Phase 2 ({label_suffix}): Training Readout...")
    
    # Readout重みとバイアス
    W_out = np.zeros((Nclasses, Nneuron))
    b_out = np.zeros(Nclasses) # 【修正】バイアス項の追加
    
    V = np.zeros(Nneuron)
    rO = np.zeros(Nneuron)
    O = 0
    k = 0

    acc_history = []
    acc_buffer = []
    
    spike_times = []
    spike_neurons = []
    
    for i in range(NumSamples):
        if i % 100 == 0:
            print(f'\r  Sample {i}/{NumSamples}', end='')

        img = X_data[i] * Gain
        label = y_data[i]
        
        target_vec = np.zeros(Nclasses)
        target_vec[label] = 1.0

        img_correct_counts = 0
        time_offset = i * Duration * dt
        
        for t in range(Duration):
            # ノイズ
            noise = 0.02 * np.random.randn(Nneuron)
            recurrent_input = np.zeros(Nneuron)
            if O == 1:
                recurrent_input = C[:, k]

            V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
            V = np.clip(V, -10, 10) # 安全策
            
            thresh_noise = 0.01 * np.random.randn(Nneuron)
            potentials = V - Thresh - thresh_noise
            k_curr = np.argmax(potentials)
            
            if potentials[k_curr] >= 0:
                O = 1
                k = k_curr
                spike_times.append(time_offset + t * dt)
                spike_neurons.append(k)
            else:
                O = 0
                
            rO = (1 - leak * dt) * rO
            if O == 1:
                rO[k] += 1.0

            # --- Readout Learning (Online Delta Rule with Bias) ---
            # y = Wx + b
            y_est_vec = np.dot(W_out, rO) + b_out
            
            pred_idx = np.argmax(y_est_vec)
            
            # 誤差逆伝播 (Delta Rule)
            error_vec = target_vec - y_est_vec
            
            # 重み更新: W += lr * err * input'
            W_out += lr_readout * np.outer(error_vec, rO)
            # バイアス更新: b += lr * err
            b_out += lr_readout * error_vec
            
            if pred_idx == label:
                img_correct_counts += 1
        
        is_correct = 1 if (img_correct_counts / Duration) > 0.5 else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0) # 移動平均の窓を200に拡大
        acc_history.append(np.mean(acc_buffer))

    print(f"\nPhase 2 Completed. Final Accuracy: {acc_history[-1]:.4f}")
    
    return acc_history, spike_times, spike_neurons

def train_readout_mnist_Rotation(F, C, X_data, y_data, Nneuron, Nx, dt, leak, Thresh, Gain, label_suffix="", NumSamples=1500, Duration=30, lr_readout=0.02):
    Nclasses = 10
    print(f"Phase 2 ({label_suffix}): Training Readout...")
    
    # Readout重みとバイアス
    W_out = np.zeros((Nclasses, Nneuron))
    b_out = np.zeros(Nclasses) # 【修正】バイアス項の追加
    
    V = np.zeros(Nneuron)
    rO = np.zeros(Nneuron)
    O = 0
    k = 0

    acc_history = []
    acc_buffer = []
    
    spike_times = []
    spike_neurons = []
    
    for i in range(NumSamples):
        if i % 100 == 0:
            print(f'\r  Sample {i}/{NumSamples}', end='')

        # 元の画像データを取得
        raw_img = X_data[i]
        
        # 250サンプル目以降なら90度回転させる
        if i >= 250:
            # (784,) -> (28, 28) に戻して回転し、再度 (784,) に平坦化
            raw_img = np.rot90(raw_img.reshape(28, 28), k=1).flatten()
            
        img = raw_img * Gain
        # === 変更点: ここまで ===
        label = y_data[i]
        
        target_vec = np.zeros(Nclasses)
        target_vec[label] = 1.0

        img_correct_counts = 0
        time_offset = i * Duration * dt
        
        for t in range(Duration):
            # ノイズ
            noise = 0.02 * np.random.randn(Nneuron)
            recurrent_input = np.zeros(Nneuron)
            if O == 1:
                recurrent_input = C[:, k]

            V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
            V = np.clip(V, -10, 10) # 安全策
            
            thresh_noise = 0.01 * np.random.randn(Nneuron)
            potentials = V - Thresh - thresh_noise
            k_curr = np.argmax(potentials)
            
            if potentials[k_curr] >= 0:
                O = 1
                k = k_curr
                spike_times.append(time_offset + t * dt)
                spike_neurons.append(k)
            else:
                O = 0
                
            rO = (1 - leak * dt) * rO
            if O == 1:
                rO[k] += 1.0

            # --- Readout Learning (Online Delta Rule with Bias) ---
            # y = Wx + b
            y_est_vec = np.dot(W_out, rO) + b_out
            
            pred_idx = np.argmax(y_est_vec)
            
            # 誤差逆伝播 (Delta Rule)
            error_vec = target_vec - y_est_vec
            
            # 重み更新: W += lr * err * input'
            W_out += lr_readout * np.outer(error_vec, rO)
            # バイアス更新: b += lr * err
            b_out += lr_readout * error_vec
            
            if pred_idx == label:
                img_correct_counts += 1
        
        is_correct = 1 if (img_correct_counts / Duration) > 0.5 else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0) # 移動平均の窓を200に拡大
        acc_history.append(np.mean(acc_buffer))

    print(f"\nPhase 2 Completed. Final Accuracy: {acc_history[-1]:.4f}")
    
    return acc_history, spike_times, spike_neurons