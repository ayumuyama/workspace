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
    
    # --- メイン学習ループ ---
    for i in range(1, TotTime):
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

def train_readout_task(F, C, Nneuron, Nx, dt, leak, Thresh):
    """
    Phase 2: 固定されたSNNを使って分類タスクを学習する
    """
    print("Phase 2: Training Readout for Classification...")

    # タスク設定
    TimeT = 30000        # 全ステップ数
    SwitchTime = 15000   # ルール変更タイミング
    lr_readout = 0.02    # 学習率
    
    W_out = np.zeros(Nneuron)
    
    # 状態変数リセット
    V = np.zeros(Nneuron)
    rO = np.zeros(Nneuron)
    O = 0
    k = 0
    
    # テスト用入力データの生成
    sigma = 30
    t_kern = np.arange(1, 1001)
    w = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t_kern - 500)**2) / (2 * sigma**2))
    w = w / np.sum(w)
    A = 2000
    
    raw_input = np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeT)
    Input = raw_input.T
    for d in range(Nx):
        Input[d, :] = A * convolve(Input[d, :], w, mode='same')

    # 記録用
    rec_target = np.zeros(TimeT)
    rec_estim = np.zeros(TimeT)
    acc_history = []
    acc_buffer = []

    for t in range(1, TimeT):
        # --- SNN更新 (学習なし) ---
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
        else:
            O = 0
            
        rO = (1 - leak * dt) * rO
        if O == 1:
            rO[k] += 1.0

        # --- タスク: 動的なターゲット ---
        inp = Input[:, t-1]
        if t < SwitchTime:
            label = 1.0 if (inp[0] + inp[1]) > 0 else -1.0 # ルールA
        else:
            label = 1.0 if (inp[0] - inp[1]) > 0 else -1.0 # ルールB (変更)

        # --- 学習 (Delta Rule) ---
        y_est = np.dot(W_out, rO)
        error = label - y_est
        W_out += lr_readout * error * rO
        
        # 記録
        rec_target[t] = label
        rec_estim[t] = y_est
        
        is_correct = 1 if np.sign(y_est) == label else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))

    return rec_target, rec_estim, acc_history, SwitchTime

if __name__ == "__main__":
    # 共通パラメータ
    Nneuron = 20
    Nx = 2
    leak = 50
    dt = 0.001
    Thresh = 0.5
    
    # 学習率パラメータ (Efficient Coding用)
    epsr = 0.001
    epsf = 0.0001
    alpha = 0.18
    beta = 1 / 0.9
    mu = 0.02 / 0.9
    
    # 1. ネットワーク構築 (Unsupervised)
    F, C = train_snn_structure(dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh)
    
    # 2. タスク学習 (Supervised / Online)
    targets, estims, accuracy, switch_t = train_readout_task(F, C, Nneuron, Nx, dt, leak, Thresh)
    
    # 3. プロット
    time_axis = np.arange(len(accuracy)) * dt
    switch_sec = switch_t * dt
    
    plt.figure(figsize=(10, 8))
    
    # 正解率
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, accuracy, label='Accuracy')
    plt.axvline(x=switch_sec, color='r', linestyle='--', label='Rule Change')
    plt.title('Classification Accuracy (Dynamic Task)')
    plt.ylabel('Moving Average Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 出力波形 (切り替え付近)
    plt.subplot(2, 1, 2)
    vis_range = range(switch_t - 500, switch_t + 500)
    plt.plot(np.array(vis_range)*dt, targets[vis_range], 'k--', alpha=0.5, label='Target')
    plt.plot(np.array(vis_range)*dt, estims[vis_range], 'g', label='Output')
    plt.axvline(x=switch_sec, color='r', linestyle='--')
    plt.title('Output vs Target (Around Rule Change)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/Brendel_Binary.png')