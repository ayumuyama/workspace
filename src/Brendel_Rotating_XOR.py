import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import os
from datetime import datetime

def train_snn_structure(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh):
    """
    Phase 1: 教師なし学習でSNNの回路(F, C)を形成する
    (元の learning 関数の分析機能を削除した軽量版)
    """
    print("Phase 1: Pre-training SNN structure...")
    
    # 学習回数 (タスクができる程度に短縮設定しています。精度が必要ならNitを増やしてください)
    Nit = 50000 
    Ntime = 1000
    TotTime = Nit * Ntime
    
    # 初期化
    # 重みはランダム初期化からスタート
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / np.sqrt(np.sum(F**2, axis=0)) # 正規化
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    
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
            print(f'{l} percent of the learning completed')
            l += 1
        # 一定間隔で入力を再生成
        if (i - 1) % Ntime == 0:
            raw_input = np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), Ntime)
            Input = raw_input.T
            for d in range(Nx):
                Input[d, :] = A * convolve(Input[d, :], w, mode='same')
        
        # 現在の入力
        curr_Input = Input[:, i % Ntime]
        
        # ノイズとリカレント入力
        noise = 0.001 * np.random.randn(Nneuron)
        recurrent_input = np.zeros(Nneuron)
        if O == 1:
            recurrent_input = C[:, k]
            
        # 膜電位更新
        V = (1 - leak * dt) * V + dt * (F.T @ curr_Input) + recurrent_input + noise
        
        # ターゲット信号の更新（学習則で使用）
        x = (1 - leak * dt) * x + dt * curr_Input
        
        # 発火判定
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
            
        # フィルタ済みスパイク列の減衰
        rO = (1 - leak * dt) * rO
        
    print("Phase 1 Completed.")
    return F, C

def train_readout_xor(F, C, Nneuron, Nx, dt, leak, Thresh):
    # 【変更1】 XORは2クラス分類なので Nclasses は 2 に固定
    Nclasses = 2
    print(f"Phase 2: Training Readout for Rotating XOR ({Nclasses}-Class)...")

    # (ここから下の初期化変数は以前と同じでOK)
    TimeT = 30000        
    SwitchTime = 15000   
    lr_readout = 0.001   
    
    W_out = np.zeros((Nclasses, Nneuron)) # 出力次元が 2 になります
    
    V = np.zeros(Nneuron)
    rO = np.zeros(Nneuron)
    O = 0
    k = 0

    spike_times = []
    spike_neurons = []
    
    # 入力データ生成
    sigma = 30
    t_kern = np.arange(1, 1001)
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_kern - 500)**2) / (2 * sigma**2))
    w = w / np.sum(w)
    A = 2000
    
    raw_input = np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeT)
    Input = raw_input.T
    for d in range(Nx):
        Input[d, :] = A * convolve(Input[d, :], w, mode='same')

    # --- 記録用配列の初期化 ---
    rec_target = np.zeros(TimeT)
    rec_estim_class = np.zeros(TimeT)
    
    # 【修正1】 全ステップのスコアを保存する配列を作成（サイズ合わせエラー防止のため）
    rec_scores = np.zeros((TimeT, Nclasses))
    
    acc_history = []
    acc_buffer = []
    
    # ... (入力データ生成などの初期化コードは省略・同じでOK) ...
    # rec_scores や acc_buffer も Nclasses=2 に合わせて初期化されます

    # --- ループ処理 ---
    for t in range(1, TimeT):
        # ... (SNN Updateの計算は全く同じなので省略) ...

        noise = 0.001 * np.random.randn(Nneuron)
        recurrent_input = np.zeros(Nneuron)
        if O == 1:
            recurrent_input = C[:, k]

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
        
        # 以前と同じ回転ロジック（ドリフト発生）
        if t >= SwitchTime:
            theta = np.pi / 2 # 90度回転
            c, s = np.cos(theta), np.sin(theta)
            rot_x = inp[0] * c - inp[1] * s
            rot_y = inp[0] * s + inp[1] * c
            inp_for_label = np.array([rot_x, rot_y])
        else:
            inp_for_label = inp

        # 【変更2】 XORのラベル決定ロジック
        # xとyの積が正なら符号が同じ（第1,3象限）→ クラス0
        # 負なら符号が違う（第2,4象限）→ クラス1
        if inp_for_label[0] * inp_for_label[1] > 0:
            label_idx = 0
        else:
            label_idx = 1

        if SwitchTime - 5 <= t <= SwitchTime + 5:
            print(f"Time: {t}, Input: {inp[:2]}, Label: {label_idx}")

        target_vec = np.zeros(Nclasses)
        target_vec[label_idx] = 1.0

        # --- Prediction & Learning ---
        y_est_vec = np.dot(W_out, rO)
        pred_idx = np.argmax(y_est_vec)
        
        # 学習などは同じ
        error_vec = target_vec - y_est_vec
        W_out += lr_readout * np.outer(error_vec, rO)
        
        # ... (記録処理も同じ) ...
        rec_target[t] = label_idx
        rec_estim_class[t] = pred_idx
        
        # 【修正2】 ここで現在のスコアベクトルをそのまま保存
        rec_scores[t, :] = y_est_vec
        
        is_correct = 1 if pred_idx == label_idx else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))

    return rec_target, rec_estim_class, acc_history, SwitchTime, spike_times, spike_neurons, rec_scores

# === メイン処理とプロット (3枚目に正解ラベル表示を追加) ===
if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存ディレクトリ
    save_dir = os.path.join('results', run_id)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # パラメータ設定
    Nneuron = 50
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

    # --- 【ここを変更】重みの保存・読み込みロジック ---
    weight_file = 'snn_structure_weights.npz' # 重み保存用ファイル名

    # ファイルが存在するか確認
    if os.path.exists(weight_file):
        print(f"Loading pre-trained weights from {weight_file}...")
        data = np.load(weight_file)
        F = data['F']
        C = data['C']
        
        # ※注意: 読み込んだ重みのサイズが現在の設定と合っているか確認（簡易チェック）
        if F.shape[1] != Nneuron:
            print("Warning: Loaded weights size mismatch! Retraining...")
            run_phase1 = True
        else:
            run_phase1 = False
            print("Phase 1 skipped.")
    else:
        print("No pre-trained weights found. Starting Phase 1...")
        run_phase1 = True

    # 必要ならPhase 1を実行
    if run_phase1:
        F, C = train_snn_structure(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh)
        # 終わったら保存
        np.savez(weight_file, F=F, C=C)
        print(f"Weights saved to {weight_file}")

    # ---------------------------------------------------
    
    # 2. タスク学習
    targets, est_classes, accuracy, switch_t, spk_t, spk_i, scores_all = \
        train_readout_xor(F, C, Nneuron, Nx, dt, leak, Thresh)
    
    # --- データ長の調整 ---
    # accuracy等は t=1 から記録されているため、長さを合わせます
    scores_valid = scores_all[1:] 
    targets_valid = targets[1:]  # 【追加】ターゲットも長さを合わせる
    
    # 時間軸を作成
    time_axis = np.arange(len(accuracy)) * dt
    switch_sec = switch_t * dt
    
    print(f"Shapes adjusted -> time: {time_axis.shape}, scores: {scores_valid.shape}, targets: {targets_valid.shape}")

    # --- 画像1: 正解率 (Accuracy) - 平滑化 ---
    def moving_average(data, window_size):
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, accuracy, color='blue', alpha=0.2, label='Raw Accuracy')
    
    window_size = 1000 
    acc_smooth = moving_average(accuracy, window_size)
    time_smooth = time_axis[window_size-1:]
    
    plt.plot(time_smooth, acc_smooth, color='blue', linewidth=2, label=f'Smoothed Accuracy')
    plt.axvline(x=switch_sec, color='r', linestyle='--', label='Rule Change', linewidth=2)
    
    plt.title(f'{Nclasses}-Class Classification Accuracy (Trend)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_Accuracy_Smoothed_Rotating_XOR.png'))
    plt.close()

    # --- 画像2: ラスタープロット (Raster) ---
    plt.figure(figsize=(12, 6))
    plt.scatter(spk_t, spk_i, s=1, c='black', marker='|', alpha=0.5)
    plt.axvline(x=switch_sec, color='r', linestyle='--', linewidth=2)
    plt.title('Spike Raster Plot (Full Range)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Neuron Index', fontsize=12)
    plt.xlim(0, time_axis[-1])
    plt.ylim(-0.5, Nneuron - 0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_Raster_Rotating_XOR.png'))
    plt.close()

    # --- 画像3: クラススコア推移 (Scores) - 正解ラベル表示付き ---
    plt.figure(figsize=(12, 6))
    
    # 拡大範囲のインデックス計算 (ルール変更前後 500ステップ)
    start_idx = max(0, switch_t - 500)
    end_idx = min(len(time_axis), switch_t + 500)
    
    # データのスライス
    time_zoom = time_axis[start_idx : end_idx]
    scores_zoom = scores_valid[start_idx : end_idx]
    targets_zoom = targets_valid[start_idx : end_idx] # 【追加】ターゲットもスライス
    
    colors = ['r', 'b']
    
    # スコアのプロット
    for c in range(Nclasses):
        plt.plot(time_zoom, scores_zoom[:, c], label=f'Score: Class {c}', color=colors[c], alpha=0.9, linewidth=2)
        
    # --- 【追加】正解ラベルの背景色表示 ---
    # グラフのY軸範囲を取得して、背景全体を塗れるようにする
    y_min, y_max = plt.ylim() 
    # 少し余裕を持たせる（見栄えのため）
    margin = (y_max - y_min) * 0.1
    y_min -= margin
    y_max += margin
    plt.ylim(y_min, y_max)

    # 正解が Class 0 の期間を薄い赤で塗りつぶし
    plt.fill_between(time_zoom, y_min, y_max, where=(targets_zoom == 0),
                     color='red', alpha=0.1, label='Target: Class 0 (Background)')
    
    # 正解が Class 1 の期間を薄い青で塗りつぶし
    plt.fill_between(time_zoom, y_min, y_max, where=(targets_zoom == 1),
                     color='blue', alpha=0.1, label='Target: Class 1 (Background)')
    # ------------------------------------

    plt.axvline(x=switch_sec, color='k', linestyle='--', linewidth=2, label='Rule Change')
    
    plt.title('Readout Scores with Ground Truth Background', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Score (Logits)', fontsize=12)
    
    # 凡例の位置調整（背景色の説明も入るため）
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_Scores_Zoom_Rotating_XOR.png'))
    plt.close()

    # --- 画像4: 正解率 (Accuracy) - 拡大版 (14.5s - 15.5s) ---
    # 拡大する時間範囲を指定
    zoom_start_time = 10.0
    zoom_end_time = 20.0

    # accuracy はリストなので、マスク処理のために NumPy 配列に変換します
    accuracy_np = np.array(accuracy)

    # この範囲に対応するデータのブールマスクを作成
    mask_raw = (time_axis >= zoom_start_time) & (time_axis <= zoom_end_time)
    mask_smooth = (time_smooth >= zoom_start_time) & (time_smooth <= zoom_end_time)

    # データを抽出 (変換した accuracy_np を使用)
    time_raw_zoom = time_axis[mask_raw]
    accuracy_zoom = accuracy_np[mask_raw]  # ここを修正しました
    time_smooth_zoom = time_smooth[mask_smooth]
    acc_smooth_zoom = acc_smooth[mask_smooth]

    plt.figure(figsize=(12, 6))
    
    # 1. 元のデータ（薄く表示）- 拡大範囲
    plt.plot(time_raw_zoom, accuracy_zoom, color='blue', alpha=0.3, label='Raw Accuracy')
    
    # 2. 平滑化したデータ（濃く表示）- 拡大範囲
    plt.plot(time_smooth_zoom, acc_smooth_zoom, color='blue', linewidth=3, label=f'Smoothed Accuracy')
    
    # ルール変更線
    plt.axvline(x=switch_sec, color='r', linestyle='--', label='Rule Change', linewidth=2)
    
    # グラフの装飾
    plt.title(f'Classification Accuracy (Zoom: {zoom_start_time}-{zoom_end_time}s)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.xlim(zoom_start_time, zoom_end_time)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, '4_Accuracy_Zoom_14.5-15.5s.png'))
    plt.close()

    print("All 4 images (Accuracy, Raster, Scores Zoom, Accuracy Zoom) saved successfully.")