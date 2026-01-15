import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# === 重み初期化 ===
def init_weights(Nx, Nneuron):
    # 重みF: 入力(Nx) -> ニューロン(Nneuron)
    F = 0.5 * np.random.randn(Nx, Nneuron)
    # 正規化 (各ニューロンへの入力重みベクトルの長さを1にする)
    F = F / (np.sqrt(np.sum(F**2, axis=0)) + 1e-8)
    
    # 重みC: ニューロン間(Nneuron x Nneuron)
    # 自己結合(対角成分)を -0.5 にしてリセット作用を持たせる
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    return F, C

# === データ読み込み ===
def load_mnist_data():
    print("Loading MNIST data...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    except:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        
    X = mnist.data
    y = mnist.target.astype(int)
    
    # 0-1に正規化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

# === Phase 1: 構造学習 (Efficient Coding) ===
def train_snn_structure_mnist(X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain):
    print("Phase 1: Pre-training SNN structure...")
    
    Nit = 50000       # 構造学習の回数
    Duration = 30    # 1枚あたりの提示時間(短縮して高速化)
    
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
        current_image = X_train[img_idx] * Gain # ゲイン適用
        
        for t in range(Duration):
            # ノイズ
            noise = 0.02 * np.random.randn(Nneuron)
            recurrent_input = np.zeros(Nneuron)
            if O == 1:
                recurrent_input = C[:, k]
                
            # 膜電位更新
            V = (1 - leak * dt) * V + dt * (F.T @ current_image) + recurrent_input + noise
            # ターゲット信号更新
            x = (1 - leak * dt) * x + dt * current_image
            
            # 発散防止クリップ
            V = np.clip(V, -10, 10)

            # 発火判定
            thresh_noise = 0.01 * np.random.randn(Nneuron)
            potentials = V - Thresh - thresh_noise
            k_curr = np.argmax(potentials)
            
            if potentials[k_curr] >= 0:
                O = 1
                k = k_curr
                total_spikes += 1
                
                # 重み更新 (ここがEfficient Codingの核)
                F[:, k] += epsf * (alpha * x - F[:, k])
                C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
                
                rO[k] += 1
            else:
                O = 0
                
            rO = (1 - leak * dt) * rO
            
    print(f"\nPhase 1 Completed. Total spikes: {total_spikes}")
    return F, C

# === Phase 2: Readout学習 (ラスタープロット機能付き) ===
def train_readout_mnist(F, C, X_data, y_data, Nneuron, Nx, dt, leak, Thresh, Gain, label_suffix=""):
    Nclasses = 10
    print(f"Phase 2 ({label_suffix}): Training Readout...")

    NumSamples = 5000  # 【修正】サンプル数を増やして学習を安定させる
    Duration = 50      
    lr_readout = 0.02  # 【修正】学習率を少し上げる
    
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

if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_MNIST_Tuned')
    save_dir = os.path.join('results', run_id)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    X_train, y_train, X_test, y_test = load_mnist_data()

    # パラメータ設定 (チューニング済み)
    Nx = 784        
    Nneuron = 1000   
    leak = 50
    dt = 0.001
    Thresh = 0.5
    
    # 【重要】Gain調整: 20だと弱すぎるので30に。クリップ処理を入れたので爆発しないはず
    Gain = 30.0     
    
    epsr = 0.0001
    epsf = 0.00001
    alpha = 0.18    
    beta = 1 / 0.9
    mu = 0.02 / 0.9

    # 重みファイル名
    weight_file = f'mnist_weights_N{Nneuron}_G{int(Gain)}.npz'
    
    if os.path.exists(weight_file):
        print(f"Loading weights from {weight_file}...")
        try:
            data = np.load(weight_file)
            F_trained = data['F']
            C_trained = data['C']
            if np.isnan(F_trained).any(): raise ValueError
        except:
            print("Weight file corrupted. Retraining...")
            F_trained, C_trained = train_snn_structure_mnist(
                X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain
            )
            np.savez(weight_file, F=F_trained, C=C_trained)
    else:
        F_trained, C_trained = train_snn_structure_mnist(
            X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain
        )
        np.savez(weight_file, F=F_trained, C=C_trained)

    # 1. Trained Model
    acc_trained, spk_t_trained, spk_i_trained = train_readout_mnist(
        F_trained, C_trained, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain, label_suffix="Trained"
    )
    
    # 2. Random Reservoir
    print("Initializing Random Reservoir...")
    F_rand, C_rand = init_weights(Nx, Nneuron)
    acc_rand, spk_t_rand, spk_i_rand = train_readout_mnist(
        F_rand, C_rand, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain, label_suffix="Random"
    )

    # --- プロット1: 正解率 ---
    plt.figure(figsize=(10, 6))
    plt.plot(acc_trained, label='Trained SNN', color='blue')
    plt.plot(acc_rand, label='Random Reservoir', color='orange', linestyle='--')
    plt.title(f'MNIST Accuracy (Full History)')
    plt.xlabel('Training Samples'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True); plt.ylim(0, 1.0)
    plt.savefig(os.path.join(save_dir, '1_Accuracy.png'))
    
    # --- プロット2: ラスタープロット (全期間) ---
    print("Plotting full raster plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 点を小さく(s=0.05)、薄く(alpha=0.3)して密度を表現
    ax1.scatter(spk_t_trained, spk_i_trained, s=0.5, c='blue', marker='.', alpha=1.0)
    ax1.set_title("Raster Plot: Trained SNN (Full Learning Process)")
    ax1.set_ylabel("Neuron Index"); ax1.set_ylim(0, Nneuron)

    ax2.scatter(spk_t_rand, spk_i_rand, s=0.5, c='blue', marker='.', alpha=1.0)
    ax2.set_title("Raster Plot: Random Reservoir (Full Learning Process)")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Neuron Index"); ax2.set_ylim(0, Nneuron)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_Full_Raster.png'))

    # --- プロット3: ラスタープロット (学習後期 拡大版) ---
    print("Plotting zoomed raster plots...")
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 最後の5秒間（約100サンプル分）を表示
    max_time = 5000 * 50 * dt # 250秒
    zoom_start = max_time - 5.0
    
    ax3.scatter(spk_t_trained, spk_i_trained, s=2, c='blue', marker='|')
    ax3.set_title("Raster Plot: Trained SNN (Last 100 Samples)")
    ax3.set_ylabel("Neuron Index"); ax3.set_ylim(0, Nneuron)
    ax3.set_xlim(zoom_start, max_time)
    
    ax4.scatter(spk_t_rand, spk_i_rand, s=2, c='blue', marker='|')
    ax4.set_title("Raster Plot: Random Reservoir (Last 100 Samples)")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Neuron Index"); ax4.set_ylim(0, Nneuron)
    ax4.set_xlim(zoom_start, max_time)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_Zoomed_Raster.png'))

    print("Done.")