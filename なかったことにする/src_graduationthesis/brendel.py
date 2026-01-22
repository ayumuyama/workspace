import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# ==========================================
# 設定パラメータ
# ==========================================
CSV_PATH = 'data/Circle/csv/BC-001.csv'   # 読み込むCSV
INPUT_COLS = [0, 1]           # 入力列
LABEL_COL = 2                 # ラベル列

# 時間・サンプル設定
STEPS_PER_SAMPLE = 30        # 1つのデータを何ステップ提示するか
PHASE1_SAMPLES = 10000         # Phase 1で学習するランダムサンプルの数
INPUT_GAIN = 5             # 入力ゲイン

# SNNハイパーパラメータ
Nneuron = 10
leak = 5
dt = 0.001
Thresh = 0.5
epsr = 0.003
epsf = 0.0003
lr_readout = 0.008
alpha = 0.5
beta = 1 / 0.9
mu = 0.02 / 0.9

# F, Cのオンライン学習　有

# ==========================================
# データ処理関数
# ==========================================
def load_raw_data(csv_path, input_cols, label_col):
    """
    CSVを読み込み、生の配列として返す (拡張はしない)
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Creating dummy XOR data...")
        N_dummy = 200
        X = np.random.randn(N_dummy, 2)
        Y = np.array([1 if x[0]*x[1] < 0 else 0 for x in X])
        df = pd.DataFrame(np.column_stack([X, Y]))
    else:
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, header=0)

    raw_inputs = df.iloc[:, input_cols].values # (Samples, Nx)
    # --- 追加: ノイズ注入 ---
    # 0.05 ~ 0.2 くらいで調整してください。大きすぎると誰も解けなくなります。
    noise_level = 0.2 
    raw_inputs += noise_level * np.random.randn(*raw_inputs.shape)
    
    raw_labels = df.iloc[:, label_col].values  # (Samples,)
    return raw_inputs, raw_labels

def expand_data(inputs, steps):
    """
    (Samples, Nx) のデータを (Nx, Samples * steps) に時間拡張する
    """
    # axis=0 (サンプル方向) に steps 回繰り返す
    # [A, B] -> [A, A, ..., A, B, B, ..., B]
    expanded = np.repeat(inputs, steps, axis=0)
    
    # ゲインをかけて転置 (Nx, Time)
    return (expanded * INPUT_GAIN).T

def prepare_phase1_input(raw_inputs, num_samples, steps):
    """
    Phase 1用にデータをランダムサンプリングして拡張する
    """
    total_available = raw_inputs.shape[0]
    
    # ランダムにインデックスを選択（重複あり）
    indices = np.random.choice(total_available, num_samples, replace=True)
    
    # 選ばれたデータを抽出
    selected_inputs = raw_inputs[indices]
    
    # 時間拡張
    input_matrix = expand_data(selected_inputs, steps)
    
    print(f"Phase 1 Data Prepared: {num_samples} random samples -> {input_matrix.shape[1]} steps")
    return input_matrix

# ==========================================
# SNN クラス・関数
# ==========================================
def run_phase1_pretraining(raw_inputs, steps_per_sample):
    """
    Phase 1: 重みの初期化と事前学習 (Unsupervised Learning)
    """
    Nx = raw_inputs.shape[1]
    
    # --- 重み初期化 ---
    print("Initializing weights...")
    # F: 入力層 -> リザーバ層
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / np.sqrt(np.sum(F**2, axis=0))
    
    # C: リカレント結合 (抑制性を含む)
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    
    # 状態変数 (Phase 1用)
    V = np.zeros(Nneuron)
    x = np.zeros(Nx)
    rO = np.zeros(Nneuron)
    Id = np.eye(Nneuron) # 行列演算用に単位行列を保持

    # --- Phase 1: 事前学習 (Random Sampling) ---
    print("\n=== Phase 1: Pre-training (Random Sampling) ===")
    
    # Phase 1用の入力データ生成 (PHASE1_SAMPLESは外部定数または引数として定義済みと仮定)
    Input_P1 = prepare_phase1_input(raw_inputs, PHASE1_SAMPLES, steps_per_sample)
    TotTime_P1 = Input_P1.shape[1]
    
    O = 0
    k = 0
    
    progress_interval = TotTime_P1 // 10
    
    for t in range(TotTime_P1):
        if progress_interval > 0 and t % progress_interval == 0:
            print(f"Phase 1: {t/TotTime_P1*100:.0f}%")
            
        curr_Input = Input_P1[:, t]
        
        # Dynamics
        noise = 0.001 * np.random.randn(Nneuron)
        recurrent = C[:, k] if O == 1 else np.zeros(Nneuron)
        
        V = (1 - leak * dt) * V + dt * (F.T @ curr_Input) + recurrent + noise
        x = (1 - leak * dt) * x + dt * curr_Input
        
        # Spike Generation
        potentials = V - Thresh - 0.01 * np.random.randn(Nneuron)
        k_curr = np.argmax(potentials)
        
        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            
            # STDP / Hebbian like update for F and C
            F[:, k] += epsf * (alpha * x - F[:, k])
            C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
            rO[k] += 1
        else:
            O = 0
        
        rO = (1 - leak * dt) * rO
        
    print("Phase 1 Completed.")
    
    # 学習済みの重みを返す
    return F, C

def run_phase2_classification(raw_inputs, raw_labels, steps_per_sample, F, C):
    """
    Phase 2: 教師あり学習と精度評価 (Supervised Learning & Evaluation)
    引数 F, C は Phase 1 で学習済みの重みを受け取る
    """
    Nx = raw_inputs.shape[1]
    
    # ラベル情報の整理
    unique_labels = np.unique(raw_labels)
    Nclasses = len(unique_labels)
    label_map = {original: idx for idx, original in enumerate(unique_labels)}
    
    # 読み出し層の初期化 (Phase 2から学習開始)
    W_out = np.zeros((Nclasses, Nneuron))
    
    # 状態変数のリセット (Phase 2開始時にクリア)
    V = np.zeros(Nneuron)
    x = np.zeros(Nx)
    rO = np.zeros(Nneuron)
    Id = np.eye(Nneuron)
    O = 0
    k = 0 # 初期スパイクニューロンインデックス
    
    # --- Phase 2: Online Classification (Sequential) ---
    print("\n=== Phase 2: Online Classification (Sequential) ===")
    
    # Phase 2用の入力データ生成
    Input_P2 = expand_data(raw_inputs, steps_per_sample)
    Labels_P2 = np.repeat(raw_labels, steps_per_sample, axis=0)
    
    TotTime_P2 = Input_P2.shape[1]
    print(f"Phase 2 Data: {raw_inputs.shape[0]} samples -> {TotTime_P2} steps")
    
    rec_acc = []
    acc_buffer = []
    sample_preds = [] # 1サンプル内の予測を貯めるリスト
    
    # スパイク記録用リスト
    spk_times = []
    spk_neurons = []
    
    progress_interval = TotTime_P2 // 10
    
    for t in range(TotTime_P2):
        if progress_interval > 0 and t % progress_interval == 0:
            print(f"Phase 2: {t/TotTime_P2*100:.0f}%")
            
        curr_Input = Input_P2[:, t]
        
        # Dynamics
        noise = 0.001 * np.random.randn(Nneuron)
        recurrent = C[:, k] if O == 1 else np.zeros(Nneuron)
        
        V = (1 - leak * dt) * V + dt * (F.T @ curr_Input) + recurrent + noise
        x = (1 - leak * dt) * x + dt * curr_Input
        
        # Spike
        potentials = V - Thresh - 0.01 * np.random.randn(Nneuron)
        k_curr = np.argmax(potentials)
        
        if potentials[k_curr] >= 0:
            O = 1
            k = k_curr
            
            # スパイク記録
            spk_times.append(t * dt)
            spk_neurons.append(k)
            
            # Phase 2でも構造学習を続ける場合 (元のコード準拠)
            F[:, k] += epsf * (alpha * x - F[:, k])
            C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
            rO[k] += 1
        else:
            O = 0
        
        rO = (1 - leak * dt) * rO
        
        # --- Readout Update (Supervised) ---
        true_label = Labels_P2[t]
        label_idx = label_map[true_label]
        
        target_vec = np.zeros(Nclasses)
        target_vec[label_idx] = 1.0
        
        y_est = np.dot(W_out, rO)
        pred_idx = np.argmax(y_est)
        
        # 学習 (Delta rule / LMS)
        error = target_vec - y_est
        W_out += lr_readout * np.outer(error, rO)
        
        # --- 評価ロジック ---
        sample_preds.append(pred_idx)
        
        # 1サンプル終了時 (steps_per_sample ごと) に判定
        if (t + 1) % steps_per_sample == 0:
            # 多数決で予測を決定
            final_pred = np.bincount(sample_preds).argmax()
            
            # 正誤判定
            true_label_sample = Labels_P2[t] 
            true_idx = label_map[true_label_sample]
            is_correct = 1 if final_pred == true_idx else 0
            
            acc_buffer.append(is_correct)
            sample_preds = [] # リセット
            
            # 移動平均計算
            window_size_samples = 500 
            if len(acc_buffer) > window_size_samples:
                acc_buffer.pop(0)
            
            rec_acc.append(np.mean(acc_buffer))

    return rec_acc, spk_times, spk_neurons

# ==========================================
# メイン実行
# ==========================================
if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results_graduationthesis', f'brendel_{run_id}')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 生データの読み込み
    raw_inputs, raw_labels = load_raw_data(CSV_PATH, INPUT_COLS, LABEL_COL)
    print(f"Raw Data Loaded: {raw_inputs.shape}")

    # 2. シミュレーション実行 (戻り値を受け取る)
    trained_F, trained_C = run_phase1_pretraining(raw_inputs=raw_inputs, steps_per_sample=STEPS_PER_SAMPLE)

    accuracy, spk_t, spk_i = run_phase2_classification(
    raw_inputs=raw_inputs, raw_labels=raw_labels, steps_per_sample=STEPS_PER_SAMPLE, F=trained_F, C=trained_C)
    
    time_axis = np.arange(len(accuracy)) * dt * STEPS_PER_SAMPLE
    
    # --- 3. 画像1: 正解率 (Accuracy) ---
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, accuracy, label='Accuracy (Moving Avg)')
    plt.title('Online Classification (Phase 1: Random Sampling / Phase 2: Sequential)')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    
    acc_path = os.path.join(save_dir, 'accuracy_plot.png')
    plt.savefig(acc_path)
    plt.close() # メモリ節約のためClose
    print(f"Done. Accuracy plot saved to {acc_path}")

    # --- 4. 【追加】画像2: ラスタープロット (Raster) ---
    plt.figure(figsize=(12, 6))
    plt.scatter(spk_t, spk_i, s=1, c='black', marker='|', alpha=0.5)
    
    plt.title('Phase 2: Spike Raster Plot', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Neuron Index', fontsize=12)
    plt.xlim(0, time_axis[-1])
    plt.ylim(-0.5, Nneuron - 0.5)
    plt.grid(True, alpha=0.3)
    
    raster_path = os.path.join(save_dir, 'raster_plot.png')
    plt.savefig(raster_path)
    plt.close()
    print(f"Done. Raster plot saved to {raster_path}")

    # データを前半と後半に分割
half_point = len(accuracy) // 2
acc_phase1 = accuracy[:half_point]
acc_phase2 = accuracy[half_point:]

# 長さが合わない場合の調整（念のため）
min_len = min(len(acc_phase1), len(acc_phase2))
acc_phase1 = acc_phase1[:min_len]
acc_phase2 = acc_phase2[:min_len]

# 時間軸（相対時間）
time_relative = np.arange(min_len) * dt * STEPS_PER_SAMPLE

plt.figure(figsize=(10, 6))

# 前半（Phase 1）のプロット
plt.plot(time_relative, acc_phase1, label='1st Cycle (0-150s)', 
         color='blue', alpha=0.5, linestyle='--')

# 後半（Phase 2）のプロット
plt.plot(time_relative, acc_phase2, label='2nd Cycle (150-300s)', 
         color='red', linewidth=2)

plt.title('Comparison of Learning Curves: 1st vs 2nd Cycle')
plt.xlabel('Relative Time (s)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

save_path_comparison = os.path.join(save_dir, 'cycle_comparison.png')
plt.savefig(save_path_comparison)
print(f"Comparison plot saved to {save_path_comparison}")