import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import os

def train_snn_structure(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh):
    """
    Phase 1: 教師なし学習でSNNの回路(F, C)を形成する
    (元の learning 関数の分析機能を削除した軽量版)
    """
    print("Phase 1: Pre-training SNN structure...")
    
    # 学習回数 (タスクができる程度に短縮設定しています。精度が必要ならNitを増やしてください)
    Nit = 12000 
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

def train_readout_multiclass(F, C, Nneuron, Nx, dt, leak, Thresh, Nclasses):
    print(f"Phase 2: Training Readout for {Nclasses}-Class Classification...")

    TimeT = 30000        
    SwitchTime = 15000   
    lr_readout = 0.02    
    
    W_out = np.zeros((Nclasses, Nneuron))
    
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

    for t in range(1, TimeT):
        # --- SNN Update ---
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

        # --- Task ---
        inp = Input[:, t-1]
        
        # 入力を回転させるための行列計算 (45度 = pi/4)
        if t >= SwitchTime:
            theta = np.pi / 4
            c, s = np.cos(theta), np.sin(theta)
            # 回転行列で入力を変換
            rot_x = inp[0] * c - inp[1] * s
            rot_y = inp[0] * s + inp[1] * c
            inp_for_label = np.array([rot_x, rot_y])
        else:
            inp_for_label = inp

        # ラベル決定ロジック（常に同じ判定基準を使うが、入力自体が回っている）
        if inp_for_label[0] >= 0 and inp_for_label[1] >= 0: label_idx = 0
        elif inp_for_label[0] < 0 and inp_for_label[1] >= 0: label_idx = 1
        elif inp_for_label[0] < 0 and inp_for_label[1] < 0: label_idx = 2
        else: label_idx = 3

        target_vec = np.zeros(Nclasses)
        target_vec[label_idx] = 1.0

        # --- Prediction & Learning ---
        y_est_vec = np.dot(W_out, rO)
        pred_idx = np.argmax(y_est_vec)
        
        error_vec = target_vec - y_est_vec
        W_out += lr_readout * np.outer(error_vec, rO)
        
        # --- 記録 ---
        rec_target[t] = label_idx
        rec_estim_class[t] = pred_idx
        
        # 【修正2】 ここで現在のスコアベクトルをそのまま保存
        rec_scores[t, :] = y_est_vec
        
        is_correct = 1 if pred_idx == label_idx else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))

    # 【修正3】 全スコア配列を返す
    return rec_target, rec_estim_class, acc_history, SwitchTime, spike_times, spike_neurons, rec_scores

# === メイン処理とプロット (3枚目だけ拡大版に戻す) ===
if __name__ == "__main__":
    import os
    
    # 保存ディレクトリ
    save_dir = 'results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # パラメータ設定
    Nneuron = 50
    Nx = 2
    Nclasses = 4
    leak = 50
    dt = 0.001
    Thresh = 0.5
    epsr = 0.001
    epsf = 0.0001
    alpha = 0.18
    beta = 1 / 0.9
    mu = 0.02 / 0.9

    # 1. 学習実行
    F, C = train_snn_structure(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh)
    
    # 2. タスク学習
    targets, est_classes, accuracy, switch_t, spk_t, spk_i, scores_all = \
        train_readout_multiclass(F, C, Nneuron, Nx, dt, leak, Thresh, Nclasses)
    
    # --- データ長の調整 (重要) ---
    # accuracy はループ内で append したため長さ 29999 (t=1 to 29999)
    # scores_all は固定長 30000 (index=0 は未使用)
    # よって、index=1以降を取り出して長さを合わせます
    scores_valid = scores_all[1:] 
    
    # 時間軸を作成 (長さ 29999)
    time_axis = np.arange(len(accuracy)) * dt
    switch_sec = switch_t * dt
    
    print(f"Shapes adjusted -> time: {time_axis.shape}, scores: {scores_valid.shape}")

    # --- 画像1: 正解率 (Accuracy) - 平滑化を追加 ---
    def moving_average(data, window_size):
        """移動平均を計算する関数"""
        # 畳み込み積分を使って高速に計算
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='valid')

    plt.figure(figsize=(12, 6))
    
    # 1. 元のデータ（薄く表示）
    # 変動を見るために alpha=0.2 で薄く残します
    plt.plot(time_axis, accuracy, color='blue', alpha=0.2, label='Raw Accuracy (0.2s window)')
    
    # 2. 平滑化したデータ（濃く表示）
    # window_size=1000 (1.0秒分) 程度にするとトレンドが見やすくなります
    window_size = 1000 
    acc_smooth = moving_average(accuracy, window_size)
    
    # convolutionを使うとデータ長が少し減るので、時間軸も合わせる必要があります
    # 'valid'モードの場合、前後が削れるので、中央に合わせて調整します
    time_smooth = time_axis[window_size-1:]
    
    plt.plot(time_smooth, acc_smooth, color='blue', linewidth=2, label=f'Smoothed Accuracy ({window_size*dt:.1f}s window)')
    
    plt.axvline(x=switch_sec, color='r', linestyle='--', label='Rule Change', linewidth=2)
    
    plt.title(f'{Nclasses}-Class Classification Accuracy (Trend)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_Accuracy_Smoothed_Nit_threetimes.png'))
    plt.close()

    # --- 画像2: ラスタープロット (Raster) - 全期間 ---
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
    plt.savefig(os.path.join(save_dir, '2_Raster_Nit_threetimes.png'))
    plt.close()

    # --- 画像3: クラススコア推移 (Scores) - ★ここだけ拡大版 ---
    plt.figure(figsize=(12, 6))
    
    # 拡大範囲のインデックス計算 (ルール変更前後 500ステップ)
    # time_axisは t=1 から始まっているので、index は t-1 に対応しますが
    # 単純に switch_t を中心としてスライスすればズレは許容範囲です
    start_idx = max(0, switch_t - 500)
    end_idx = min(len(time_axis), switch_t + 500)
    
    # 同じ範囲でスライス (ここが重要)
    time_zoom = time_axis[start_idx : end_idx]
    scores_zoom = scores_valid[start_idx : end_idx]
    
    colors = ['r', 'g', 'b', 'orange']
    for c in range(Nclasses):
        plt.plot(time_zoom, scores_zoom[:, c], label=f'Class {c}', color=colors[c], alpha=0.8, linewidth=2)
        
    plt.axvline(x=switch_sec, color='k', linestyle='--', linewidth=2, label='Rule Change')
    plt.title('Readout Class Scores (Zoom: Switch ±0.5s)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Score (Logits)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_Scores_Zoom_Nit_threetimes.png'))
    plt.close()

    print("All images (Accuracy, Raster, Zoomed Scores) saved successfully.")