import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# ==========================================
# 設定パラメータ
# ==========================================
CSV_PATH = 'data/BC-001.csv'   # 読み込むCSV
INPUT_COLS = [0, 1]           # 入力列
LABEL_COL = 2                 # ラベル列

# 時間・サンプル設定
STEPS_PER_SAMPLE = 30        # 1つのデータを何ステップ提示するか
PHASE1_SAMPLES = 10000         # Phase 1で学習するランダムサンプルの数
INPUT_GAIN = 5             # 入力ゲイン

# SNNハイパーパラメータ
Nneuron = 50
leak = 5
dt = 0.001
Thresh = 0.5
epsr = 0.005
epsf = 0.0005
lr_readout = 0.05
alpha = 0.18
beta = 1 / 0.9
mu = 0.02 / 0.9

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
def run_snn_simulation(raw_inputs, raw_labels, steps_per_sample):
    
    Nx = raw_inputs.shape[1]
    
    # --- 重み初期化 ---
    print("Initializing weights...")
    F = 0.5 * np.random.randn(Nx, Nneuron)
    F = F / np.sqrt(np.sum(F**2, axis=0))
    C = -0.2 * np.random.rand(Nneuron, Nneuron) - 0.5 * np.eye(Nneuron)
    
    # ラベル情報の整理
    unique_labels = np.unique(raw_labels)
    Nclasses = len(unique_labels)
    label_map = {original: idx for idx, original in enumerate(unique_labels)}
    W_out = np.zeros((Nclasses, Nneuron))
    
    # 状態変数
    V = np.zeros(Nneuron)
    x = np.zeros(Nx)
    rO = np.zeros(Nneuron)
    Id = np.eye(Nneuron)

    # ---------------------------------------------------------
    # Phase 1: 事前学習 (Random Sampling)
    # ---------------------------------------------------------
    print("\n=== Phase 1: Pre-training (Random 5000 Samples) ===")
    
    # Phase 1用の入力データ生成
    Input_P1 = prepare_phase1_input(raw_inputs, PHASE1_SAMPLES, steps_per_sample)
    TotTime_P1 = Input_P1.shape[1]
    
    O = 0
    k = 0
    
    progress_interval = TotTime_P1 // 10
    
    for t in range(TotTime_P1):
        if t % progress_interval == 0:
            print(f"Phase 1: {t/TotTime_P1*100:.0f}%")
            
        curr_Input = Input_P1[:, t]
        
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
            # Update F, C
            F[:, k] += epsf * (alpha * x - F[:, k])
            C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
            rO[k] += 1
        else:
            O = 0
        
        rO = (1 - leak * dt) * rO
        
    print("Phase 1 Completed.")

    # ---------------------------------------------------------
    # Phase 2: 分類タスク (Sequential Step Input)
    # ---------------------------------------------------------
    print("\n=== Phase 2: Online Classification (Sequential) ===")
    
    # Phase 2用の入力データ生成（CSVの順番通り）
    Input_P2 = expand_data(raw_inputs, steps_per_sample)
    Labels_P2 = np.repeat(raw_labels, steps_per_sample, axis=0)
    
    TotTime_P2 = Input_P2.shape[1]
    print(f"Phase 2 Data: {raw_inputs.shape[0]} samples -> {TotTime_P2} steps")
    
    # 状態リセット
    V = np.zeros(Nneuron)
    x = np.zeros(Nx)
    rO = np.zeros(Nneuron)
    O = 0
    
    rec_acc = []
    acc_buffer = []
    
    # --- 【追加】スパイク記録用リスト ---
    spk_times = []
    spk_neurons = []
    
    progress_interval = TotTime_P2 // 10
    
    for t in range(TotTime_P2):
        if t % progress_interval == 0:
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
            
            # --- 【追加】スパイク記録 ---
            spk_times.append(t * dt)
            spk_neurons.append(k)
            
            # Phase 2 Update (Structure)
            # F[:, k] += epsf * (alpha * x - F[:, k])
            # C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
            rO[k] += 1
        else:
            O = 0
        
        rO = (1 - leak * dt) * rO
        
        # Readout Update
        true_label = Labels_P2[t]
        label_idx = label_map[true_label]
        target_vec = np.zeros(Nclasses)
        target_vec[label_idx] = 1.0
        
        y_est = np.dot(W_out, rO)
        pred_idx = np.argmax(y_est)
        
        error = target_vec - y_est
        W_out += lr_readout * np.outer(error, rO)
        
        # Accuracy Tracking
        is_correct = 1 if pred_idx == label_idx else 0
        acc_buffer.append(is_correct)
        
        # 移動平均窓 (10サンプル分くらいの期間)
        window_size = STEPS_PER_SAMPLE * 10
        if len(acc_buffer) > window_size:
            acc_buffer.pop(0)
        rec_acc.append(np.mean(acc_buffer))

    # --- 【追加】スパイク情報も返す ---
    return rec_acc, spk_times, spk_neurons

# ==========================================
# メイン実行
# ==========================================
if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results_graduationthesis', f'csv_random_phase1_{run_id}')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 生データの読み込み
    raw_inputs, raw_labels = load_raw_data(CSV_PATH, INPUT_COLS, LABEL_COL)
    print(f"Raw Data Loaded: {raw_inputs.shape}")

    # 2. シミュレーション実行 (戻り値を受け取る)
    accuracy, spk_t, spk_i = run_snn_simulation(raw_inputs, raw_labels, STEPS_PER_SAMPLE)
    
    time_axis = np.arange(len(accuracy)) * dt
    
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