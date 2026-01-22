import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import os
from datetime import datetime

# === 重み初期化関数 (共通化) ===
def init_weights(Nx, Nneuron):
    """
    SNNの重み F, C をランダムに初期化して返す
    """
    # 重みはランダム初期化からスタート
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / np.sqrt(np.sum(F**2, axis=0)) # 正規化
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    return F, C

def train_snn_structure(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh):
    """
    Phase 1: 教師なし学習でSNNの回路(F, C)を形成する
    """
    print("Phase 1: Pre-training SNN structure...")
    
    # 学習回数
    Nit = 50000 
    Ntime = 1000
    TotTime = Nit * Ntime
    
    # --- 初期化 (共通関数を使用) ---
    F, C = init_weights(Nx, Nneuron)
    
    V = np.zeros(Nneuron)
    O = 0
    k = 0
    rO = np.zeros(Nneuron)
    x = np.zeros(Nx)
    Id = np.eye(Nneuron)
    
    # 入力生成用パラメータ
    A = 2000
    sigma = 30
    t_kern = np.arange(1, 1001)
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_kern - 500)**2) / (2 * sigma**2))
    w = w / np.sum(w)
    
    Input = np.zeros((Nx, Ntime))
    l = 1

    # --- メイン学習ループ ---
    for i in range(1, TotTime):
        # 進捗表示
        if (i / TotTime) > (l / 100.0):
            print(f'\rPhase 1 Progress: {l}%', end='')
            l += 1
        
        # 一定間隔で入力を再生成
        if (i - 1) % Ntime == 0:
            raw_input = np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), Ntime)
            Input = raw_input.T
            for d in range(Nx):
                Input[d, :] = A * convolve(Input[d, :], w, mode='same')
        
        curr_Input = Input[:, i % Ntime]
        
        noise = 0.001 * np.random.randn(Nneuron)
        recurrent_input = np.zeros(Nneuron)
        if O == 1:
            recurrent_input = C[:, k]
            
        V = (1 - leak * dt) * V + dt * (F.T @ curr_Input) + recurrent_input + noise
        x = (1 - leak * dt) * x + dt * curr_Input
        
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        k_curr = np.argmax(potentials)
        
        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            
            # === 重み更新 (Efficient Coding) ===
            F[:, k] += epsf * (alpha * x - F[:, k])
            C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
            
            rO[k] += 1
        else:
            O = 0
            
        rO = (1 - leak * dt) * rO
        
    print("\nPhase 1 Completed.")
    return F, C

def train_readout_xor(F, C, Nneuron, Nx, dt, leak, Thresh, label_suffix=""):
    """
    Phase 2: Readout学習 (引数に label_suffix を追加してログを区別)
    """
    Nclasses = 2
    print(f"Phase 2 ({label_suffix}): Training Readout for Rotating XOR...")

    TimeT = 30000        
    SwitchTime = 15000   
    lr_readout = 0.001  
    
    W_out = np.zeros((Nclasses, Nneuron))
    
    V = np.zeros(Nneuron)
    rO = np.zeros(Nneuron)
    O = 0
    k = 0

    spike_times = []
    spike_neurons = []
    
    sigma = 30
    t_kern = np.arange(1, 1001)
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_kern - 500)**2) / (2 * sigma**2))
    w = w / np.sum(w)
    A = 2000
    
    # 共通の乱数シードを使用したい場合もありますが、
    # ここではランダム生成をそのまま実行します（統計的に同様のタスク）
    raw_input = np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeT)
    Input = raw_input.T
    for d in range(Nx):
        Input[d, :] = A * convolve(Input[d, :], w, mode='same')

    rec_target = np.zeros(TimeT)
    rec_estim_class = np.zeros(TimeT)
    rec_scores = np.zeros((TimeT, Nclasses))
    
    acc_history = []
    acc_buffer = []
    
    # --- ループ処理 ---
    for t in range(1, TimeT):
        noise = 0.001 * np.random.randn(Nneuron)
        recurrent_input = np.zeros(Nneuron)
        if O == 1:
            recurrent_input = C[:, k]

        # SNN Dynamics (F, C は固定)
        V = (1 - leak * dt) * V + dt * (F.T @ Input[:, t-1]) + recurrent_input + noise
        
        thresh_noise = 0.01 * np.random.randn(Nneuron)
        potentials = V - Thresh - thresh_noise
        k_curr = np.argmax(potentials)
        
        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            spike_times.append(t * dt)
            spike_neurons.append(k)
        else:
            O = 0
            
        rO = (1 - leak * dt) * rO
        if O == 1:
            rO[k] += 1.0

        # --- Task: Rotating XOR Logic ---
        inp = Input[:, t-1]
        
        if t >= SwitchTime:
            theta = np.pi / 2
            c_rot, s_rot = np.cos(theta), np.sin(theta)
            rot_x = inp[0] * c_rot - inp[1] * s_rot
            rot_y = inp[0] * s_rot + inp[1] * c_rot
            inp_for_label = np.array([rot_x, rot_y])
        else:
            inp_for_label = inp

        if inp_for_label[0] * inp_for_label[1] > 0:
            label_idx = 0
        else:
            label_idx = 1

        target_vec = np.zeros(Nclasses)
        target_vec[label_idx] = 1.0

        # --- Prediction & Learning (Readout Only) ---
        y_est_vec = np.dot(W_out, rO)
        pred_idx = np.argmax(y_est_vec)
        
        error_vec = target_vec - y_est_vec
        W_out += lr_readout * np.outer(error_vec, rO)
        
        rec_target[t] = label_idx
        rec_estim_class[t] = pred_idx
        rec_scores[t, :] = y_est_vec
        
        is_correct = 1 if pred_idx == label_idx else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))

    return rec_target, rec_estim_class, acc_history, SwitchTime, spike_times, spike_neurons, rec_scores

# === メイン処理 ===
if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', run_id)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # パラメータ設定
    Nneuron = 10
    Nx = 2
    Nclasses = 2
    leak = 50
    dt = 0.001
    Thresh = 0.5
    epsr = 0.001
    epsf = 0.0001
    alpha = 0.18
    beta = 1 / 0.9
    mu = 0.02 / 0.9

    # ----------------------------------------------------------------
    # 1. Trained Model (Efficient Coding) の準備
    # ----------------------------------------------------------------
    weight_file = 'snn_structure_weights.npz'
    
    if os.path.exists(weight_file):
        print(f"Loading pre-trained weights from {weight_file}...")
        data = np.load(weight_file)
        F_trained = data['F']
        C_trained = data['C']
        
        if F_trained.shape[1] != Nneuron:
            print("Warning: Size mismatch! Retraining...")
            run_phase1 = True
        else:
            run_phase1 = False
            print("Phase 1 skipped (Loaded).")
    else:
        print("No pre-trained weights found. Starting Phase 1...")
        run_phase1 = True

    if run_phase1:
        F_trained, C_trained = train_snn_structure(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh)
        np.savez(weight_file, F=F_trained, C=C_trained)
        print(f"Weights saved to {weight_file}")

    # ----------------------------------------------------------------
    # 2. Random Reservoir (Untrained) の準備
    # ----------------------------------------------------------------
    print("Initializing Random Reservoir weights (No pre-training)...")
    F_rand, C_rand = init_weights(Nx, Nneuron)

    # ----------------------------------------------------------------
    # 3. Readout 学習の実行 (両モデル)
    # ----------------------------------------------------------------
    
    # (A) Trained Model
    print("\n--- Running Trained Model ---")
    targets, _, acc_trained, switch_t, spk_t, spk_i, scores_trained = \
        train_readout_xor(F_trained, C_trained, Nneuron, Nx, dt, leak, Thresh, label_suffix="Trained")
    
    # (B) Random Reservoir
    print("\n--- Running Random Reservoir ---")
    _, _, acc_rand, _, spk_t_rand, spk_i_rand, _ = \
        train_readout_xor(F_rand, C_rand, Nneuron, Nx, dt, leak, Thresh, label_suffix="Random")

    # --- データ整形 ---
    # accuracy等は t=1 から記録されているため調整
    time_axis = np.arange(len(acc_trained)) * dt
    switch_sec = switch_t * dt
    
    scores_valid = scores_trained[1:]
    targets_valid = targets[1:]

    # 平滑化関数
    def moving_average(data, window_size):
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='valid')
    
    window_size = 1000
    time_smooth = time_axis[window_size-1:]
    
    # それぞれ平滑化
    acc_smooth_trained = moving_average(acc_trained, window_size)
    acc_smooth_rand = moving_average(acc_rand, window_size)

    print(f"Plotting results...")

    # ================================================================
    # 画像1: 正解率比較 (Trained vs Random)
    # ================================================================
    plt.figure(figsize=(12, 6))
    
    # Trained (Blue)
    plt.plot(time_smooth, acc_smooth_trained, color='blue', linewidth=2, label='Trained SNN (Efficient Coding)')
    plt.plot(time_axis, acc_trained, color='blue', alpha=0.1) # Raw data background
    
    # Random (Orange)
    plt.plot(time_smooth, acc_smooth_rand, color='orange', linewidth=2, label='Random Reservoir')
    plt.plot(time_axis, acc_rand, color='orange', alpha=0.1) # Raw data background

    plt.axvline(x=switch_sec, color='r', linestyle='--', label='Rule Change', linewidth=2)
    
    plt.title(f'Performance Comparison: Trained vs Random Reservoir', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_Accuracy_Comparison.png'))
    plt.close()

    # ================================================================
    # 画像2: ラスタープロット (Trained Modelのみ)
    # ================================================================
    plt.figure(figsize=(12, 6))
    plt.scatter(spk_t, spk_i, s=1, c='black', marker='|', alpha=0.5)
    plt.axvline(x=switch_sec, color='r', linestyle='--', linewidth=2)
    plt.title('Spike Raster Plot (Trained Model)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Neuron Index', fontsize=12)
    plt.xlim(0, time_axis[-1])
    plt.ylim(-0.5, Nneuron - 0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_Raster_Trained.png'))
    plt.close()

    # ================================================================
    # 画像3: クラススコア推移 (Trained Modelのみ) - 正解背景あり
    # ================================================================
    plt.figure(figsize=(12, 6))
    start_idx = max(0, switch_t - 500)
    end_idx = min(len(time_axis), switch_t + 500)
    
    time_zoom = time_axis[start_idx : end_idx]
    scores_zoom = scores_valid[start_idx : end_idx]
    targets_zoom = targets_valid[start_idx : end_idx]
    
    colors = ['r', 'b']
    for c in range(Nclasses):
        plt.plot(time_zoom, scores_zoom[:, c], label=f'Score: Class {c}', color=colors[c], alpha=0.9, linewidth=2)
        
    y_min, y_max = plt.ylim() 
    margin = (y_max - y_min) * 0.1
    y_min -= margin; y_max += margin
    plt.ylim(y_min, y_max)

    plt.fill_between(time_zoom, y_min, y_max, where=(targets_zoom == 0), color='red', alpha=0.1, label='Target: Class 0')
    plt.fill_between(time_zoom, y_min, y_max, where=(targets_zoom == 1), color='blue', alpha=0.1, label='Target: Class 1')
    
    plt.axvline(x=switch_sec, color='k', linestyle='--', linewidth=2, label='Rule Change')
    plt.title('Readout Scores (Trained Model) around Rule Change', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Score (Logits)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_Scores_Zoom_Trained.png'))
    plt.close()

    # ================================================================
    # 画像4: 拡大比較 (Accuracy Zoom)
    # ================================================================
    zoom_start_time = 10.0
    zoom_end_time = 20.0
    
    # マスク作成
    mask_smooth = (time_smooth >= zoom_start_time) & (time_smooth <= zoom_end_time)
    
    time_smooth_zoom = time_smooth[mask_smooth]
    acc_trained_zoom = acc_smooth_trained[mask_smooth]
    acc_rand_zoom = acc_smooth_rand[mask_smooth]

    plt.figure(figsize=(12, 6))
    
    # Trained
    plt.plot(time_smooth_zoom, acc_trained_zoom, color='blue', linewidth=3, label='Trained SNN')
    # Random
    plt.plot(time_smooth_zoom, acc_rand_zoom, color='orange', linewidth=3, label='Random Reservoir', linestyle='-')

    plt.axvline(x=switch_sec, color='r', linestyle='--', label='Rule Change', linewidth=2)
    
    plt.title(f'Accuracy Comparison (Zoom: {zoom_start_time}-{zoom_end_time}s)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.xlim(zoom_start_time, zoom_end_time)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_Accuracy_Comparison_Zoom.png'))
    plt.close()

    # ================================================================
    # 画像2-B: ラスタープロット (Random Reservoir) 【追加】
    # ================================================================
    plt.figure(figsize=(12, 6))
    # ランダムリザバーのスパイクデータ (spk_t_rand, spk_i_rand) を使用
    plt.scatter(spk_t_rand, spk_i_rand, s=1, c='black', marker='|', alpha=0.5)
    
    # ルール変更線
    plt.axvline(x=switch_sec, color='r', linestyle='--', linewidth=2)
    
    plt.title('Spike Raster Plot (Random Reservoir)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Neuron Index', fontsize=12)
    plt.xlim(0, time_axis[-1])
    plt.ylim(-0.5, Nneuron - 0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # ファイル名を変更して保存
    plt.savefig(os.path.join(save_dir, '5_Raster_Random.png'))
    plt.close()
    
    print("All images saved successfully.")