import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

def train_snn_structure(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh):
    """
    Phase 1: 教師なし学習でSNNの回路(F, C)を形成する
    (元の learning 関数の分析機能を削除した軽量版)
    """
    print("Phase 1: Pre-training SNN structure...")
    
    # 学習回数 (タスクができる程度に短縮設定しています。精度が必要ならNitを増やしてください)
    Nit = 4000 
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
        
        if t < SwitchTime:
            # Rule A
            if inp[0] >= 0 and inp[1] >= 0: label_idx = 0
            elif inp[0] < 0 and inp[1] >= 0: label_idx = 1
            elif inp[0] < 0 and inp[1] < 0: label_idx = 2
            else: label_idx = 3
        else:
            # Rule B (Shifted)
            if inp[0] >= 0 and inp[1] >= 0: label_idx = 1
            elif inp[0] < 0 and inp[1] >= 0: label_idx = 2
            elif inp[0] < 0 and inp[1] < 0: label_idx = 3
            else: label_idx = 0

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

if __name__ == "__main__":
    # ... (パラメータ設定やPhase 1は変更なし) ...
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

    # Phase 1
    F, C = train_snn_structure(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh)
    
    # Phase 2
    # 戻り値の名前を scores_all に変更
    targets, est_classes, accuracy, switch_t, spk_t, spk_i, scores_all = \
        train_readout_multiclass(F, C, Nneuron, Nx, dt, leak, Thresh, Nclasses)
    
    # Phase 3 Plotting
    time_axis = np.arange(len(accuracy)) * dt
    switch_sec = switch_t * dt
    
    plt.figure(figsize=(10, 12))
    
    # 1. Accuracy
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, accuracy, label='Accuracy', color='blue')
    plt.axvline(x=switch_sec, color='r', linestyle='--', label='Rule Change')
    plt.title(f'{Nclasses}-Class Classification Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)

    # 2. Raster
    plt.subplot(3, 1, 2)
    plt.scatter(spk_t, spk_i, s=1, c='black', marker='|')
    plt.axvline(x=switch_sec, color='r', linestyle='--')
    plt.title('Spike Raster Plot')
    plt.xlim(0, time_axis[-1])
    plt.ylim(-0.5, Nneuron - 0.5)
    plt.grid(True, alpha=0.3)

    # 3. Class Scores (Zoom)
    plt.subplot(3, 1, 3)
    
    # 【修正4】 インデックス範囲を指定して、時間軸とデータを同じスライスで切り出す
    start_idx = switch_t - 500
    end_idx = switch_t + 500
    
    # numpyスライスを使うことで shape が確実に一致します
    # time_axis[start:end] -> shape (1000,)
    # scores_all[start:end] -> shape (1000, 4)
    time_zoom = time_axis[start_idx : end_idx]
    scores_zoom = scores_all[start_idx : end_idx]
    
    colors = ['r', 'g', 'b', 'orange']
    for c in range(Nclasses):
        plt.plot(time_zoom, scores_zoom[:, c], label=f'Class {c}', color=colors[c], alpha=0.8)
        
    plt.axvline(x=switch_sec, color='k', linestyle='--', alpha=0.5)
    plt.title('Class Scores (Zoom around Rule Change)')
    plt.xlabel('Time (s)')
    plt.ylabel('Readout Activity')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/SNN_Multiclass_Fixed.png')