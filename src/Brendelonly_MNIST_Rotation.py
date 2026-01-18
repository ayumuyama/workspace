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

    NumSamples = 1500  # 【修正】サンプル数を増やして学習を安定させる
    Duration = 30      
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

# === Phase 2b: Poisson Baseline 学習 ===
# def train_poisson_readout_mnist(F, C, X_data, y_data, Nneuron, Nx, dt, leak, Thresh, Gain):
#     """
#     提案モデル(SNN)と同じ発火率を持つが、タイミングがランダムなポアソンネットワークでReadoutを学習する。
#     1. SNNを実行して、その画像に対する各ニューロンの発火率(レート)を計測する。
#     2. そのレートに基づいて、ポアソン過程でランダムにスパイクを生成する。
#     3. そのランダムなスパイク列を使ってReadout(W_out)を学習させる。
#     """
#     Nclasses = 10
#     print(f"Phase 2 (Poisson Baseline): Training Readout with Rate-matched Poisson Spikes...")

#     NumSamples = 500  # Phase 2と同じ設定
#     Duration = 30
#     lr_readout = 0.02
    
#     # Poisson用 Readout重みとバイアス
#     W_out_p = np.zeros((Nclasses, Nneuron))
#     b_out_p = np.zeros(Nclasses)
    
#     acc_history = []
#     acc_buffer = []
    
#     spike_times = []
#     spike_neurons = []
    
#     # SNNの状態変数 (レート計測用)
#     V = np.zeros(Nneuron)
#     rO_snn = np.zeros(Nneuron) # SNN内部のフィルタ値（リセット用）
    
#     for i in range(NumSamples):
#         if i % 100 == 0:
#             print(f'\r  Poisson Sample {i}/{NumSamples}', end='')

#         raw_img = X_data[i]
        
#         if i >= 250:
#             # 90度回転 (反時計回り)
#             raw_img = np.rot90(raw_img.reshape(28, 28), k=1).flatten()
            
#         img = raw_img * Gain
#         label = y_data[i]
#         target_vec = np.zeros(Nclasses)
#         target_vec[label] = 1.0

#         # --- Step 1: SNNを走らせて「ターゲット発火率」を取得する ---
#         # 重み F, C は学習済みのものを固定して使う
#         # ここでは学習(更新)は行わず、スパイク数だけをカウントする
        
#         # SNNの状態リセット
#         V[:] = 0
#         rO_snn[:] = 0
#         O = 0
#         k = 0
        
#         # 各ニューロンがこの画像に対して何発撃ったかカウント
#         spike_counts = np.zeros(Nneuron)
        
#         # SNN Forward Pass (レート取得用)
#         for t in range(Duration):
#             noise = 0.02 * np.random.randn(Nneuron)
#             recurrent_input = np.zeros(Nneuron)
#             if O == 1:
#                 recurrent_input = C[:, k]

#             V = (1 - leak * dt) * V + dt * (F.T @ img) + recurrent_input + noise
#             V = np.clip(V, -10, 10)
            
#             thresh_noise = 0.01 * np.random.randn(Nneuron)
#             potentials = V - Thresh - thresh_noise
#             k_curr = np.argmax(potentials)
            
#             if potentials[k_curr] >= 0:
#                 O = 1
#                 k = k_curr
#                 spike_counts[k] += 1
                
#                 # リセット用rOの更新(ダイナミクス維持のため)
#                 rO_snn[k] += 1
#             else:
#                 O = 0
            
#             rO_snn = (1 - leak * dt) * rO_snn

#         # --- Step 2 & 3: スパイク時刻のシャッフルとReadout学習 ---
        
#         # 事前にこのサンプルの全タイムステップ分のスパイクパタンを作成する
#         # shape: (Duration, Nneuron)
#         random_spike_train = np.zeros((Duration, Nneuron))
        
#         for n_idx in range(Nneuron):
#             n_spikes = int(spike_counts[n_idx])
            
#             if n_spikes > 0:
#                 # Durationの範囲内で、n_spikes個のユニークな時刻をランダムに選ぶ
#                 # replace=False にすることで、同じ時刻に重なるのを防ぐ（バイナリスパイクの維持）
#                 # ※もし n_spikes > Duration になるほど高頻度なら replace=True か min(n, Duration) が必要だが、WTAならFalseでOK
#                 if n_spikes > Duration: n_spikes = Duration 
                
#                 random_times = np.random.choice(Duration, n_spikes, replace=False)
#                 random_spike_train[random_times, n_idx] = 1.0
        
#         # Poissonネットワーク用の状態変数
#         rO_poisson = np.zeros(Nneuron)
#         img_correct_counts = 0
#         time_offset = i * Duration * dt

#         # 時間ループ
#         for t in range(Duration):
#             # シャッフルされたスパイクを取り出す
#             spikes = random_spike_train[t]
            
#             # フィルタ更新
#             rO_poisson = (1 - leak * dt) * rO_poisson + spikes
            
#             # 記録用 (ラスタープロット)
#             active_neurons = np.where(spikes > 0)[0]
#             for neuron_idx in active_neurons:
#                 spike_times.append(time_offset + t * dt)
#                 spike_neurons.append(neuron_idx)

#             # --- Readout Learning (Delta Rule) ---
#             # ポアソンスパイクのフィルタ値を使って学習
#             y_est_vec = np.dot(W_out_p, rO_poisson) + b_out_p
#             pred_idx = np.argmax(y_est_vec)
            
#             error_vec = target_vec - y_est_vec
            
#             # 重み更新
#             W_out_p += lr_readout * np.outer(error_vec, rO_poisson)
#             b_out_p += lr_readout * error_vec
            
#             if pred_idx == label:
#                 img_correct_counts += 1
        
#         is_correct = 1 if (img_correct_counts / Duration) > 0.5 else 0
#         acc_buffer.append(is_correct)
#         if len(acc_buffer) > 200: acc_buffer.pop(0)
#         acc_history.append(np.mean(acc_buffer))

#     print(f"\nPhase 2 (Poisson) Completed. Final Accuracy: {acc_history[-1]:.4f}")
#     return acc_history, spike_times, spike_neurons

if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_MNIST_Tuned')
    save_dir = os.path.join('results', run_id)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    X_train, y_train, X_test, y_test = load_mnist_data()

    # パラメータ設定 (チューニング済み)
    Nx = 784        
    Nneuron = 5000   
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
    weight_file = f'mnist_weights_N{Nneuron}_G{int(Gain)}_Brendelonly.npz'
    
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
    
    # 2. Independent Poisson Baseline (New!)
    # ランダムな重みではなく、学習済みSNNと同じ発火率を持つポアソンモデルと比較する
    # acc_poisson, spk_t_poisson, spk_i_poisson = train_poisson_readout_mnist(
    #     F_trained, C_trained, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain
    # )

    # --- プロット1: 正解率 ---
    plt.figure(figsize=(10, 6))
    plt.plot(acc_trained, label='Trained SNN (Precise Timing)', color='blue')
    plt.axvline(x=250, color='red', linestyle='--', label='Input Change')# plt.plot(acc_poisson, label='Poisson Baseline (Rate Matched)', color='green', linestyle='--')
    plt.title(f'Accuracy: Precise Timing vs Rate Coding')
    plt.xlabel('Training Samples'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True); plt.ylim(0, 1.0)
    plt.savefig(os.path.join(save_dir, '1_Accuracy_Comparison.png'))
    
    # --- プロット2: ラスタープロット (拡大版) ---
    print("Plotting raster plot...")
    
    # subplotsを単一のものに変更し、高さ(figsize)を適宜調整しています
    fig, ax = plt.subplots(figsize=(15, 5)) 
    
    max_time = 500 * 30 * dt

    ax.scatter(spk_t_trained, spk_i_trained, s=5, c='blue', marker='|')
    ax.axvline(x=250, color='red', linestyle='--', linewidth=1.5, label='Input Change')
    ax.set_title("Trained SNN: Structured & Sparse")
    ax.set_ylabel("Neuron Index")
    ax.set_ylim(0, Nneuron)
    ax.set_xlim(0, max_time)
    
    # 元々ax2にあったX軸ラベルをこちらに設定
    ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_Raster_Comparison.png'))

    print("Done.")