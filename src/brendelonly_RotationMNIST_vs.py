import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys

# パス設定と自作モジュールの読み込み
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import brendel_utils as bu

# === 新規作成: 途中からFとCも再学習する関数 ===
def train_readout_mnist_Rotation_Retrain(F, C, X_data, y_data, Nneuron, Nx, dt, leak, Thresh, Gain,
                                         epsf, epsr, alpha, beta, mu,
                                         label_suffix="", NumSamples=1500, Duration=30, lr_readout=0.02,
                                         rotation_start_idx=250, retrain_start_idx=300):
    """
    300サンプル目以降でFとCもオンライン学習する関数
    rotation_start_idx: 入力画像が回転し始めるサンプルインデックス (デフォルト: 250)
    retrain_start_idx: FとCの学習を開始するサンプルインデックス (デフォルト: 300)
    """
    Nclasses = 10
    print(f"Phase 2 ({label_suffix}): Training Readout (with Retraining from sample {retrain_start_idx})...")
    
    # Readout重みとバイアス
    W_out = np.zeros((Nclasses, Nneuron))
    b_out = np.zeros(Nclasses)
    
    # 状態変数の初期化
    V = np.zeros(Nneuron)
    x = np.zeros(Nx) # 入力トレース (Fの学習に必要)
    rO = np.zeros(Nneuron)
    O = 0
    k = 0
    
    Id = np.eye(Nneuron)

    acc_history = []
    acc_buffer = []
    
    spike_times = []
    spike_neurons = []
    
    # 重みのコピー（元の変数を書き換えないため）
    F_curr = F.copy()
    C_curr = C.copy()

    for i in range(NumSamples):
        if i % 100 == 0:
            print(f'\r  Sample {i}/{NumSamples}', end='')

        # 元の画像データを取得
        raw_img = X_data[i]
        
        # 指定サンプル以降なら90度回転
        if i >= rotation_start_idx:
            # (784,) -> (28, 28) に戻して回転し、再度 (784,) に平坦化
            raw_img = np.rot90(raw_img.reshape(28, 28), k=1).flatten()
            
        img = raw_img * Gain
        label = y_data[i]
        
        target_vec = np.zeros(Nclasses)
        target_vec[label] = 1.0

        img_correct_counts = 0
        time_offset = i * Duration * dt
        
        # 再学習が有効かどうかのフラグ
        enable_retrain = (i >= retrain_start_idx)
        
        for t in range(Duration):
            # ノイズ
            noise = 0.02 * np.random.randn(Nneuron)
            recurrent_input = np.zeros(Nneuron)
            if O == 1:
                recurrent_input = C_curr[:, k]

            # 膜電位の更新
            V = (1 - leak * dt) * V + dt * (F_curr.T @ img) + recurrent_input + noise
            # 入力トレースの更新 (Fの学習用)
            x = (1 - leak * dt) * x + dt * img
            
            V = np.clip(V, -10, 10) 
            
            thresh_noise = 0.01 * np.random.randn(Nneuron)
            potentials = V - Thresh - thresh_noise
            k_curr = np.argmax(potentials)
            
            if potentials[k_curr] >= 0:
                O = 1
                k = k_curr
                spike_times.append(time_offset + t * dt)
                spike_neurons.append(k)
                
                # === ここで F と C を更新 (指定サンプル以降のみ) ===
                if enable_retrain:
                    # Fの更新: F[:, k] += epsf * (alpha * x - F[:, k])
                    F_curr[:, k] += epsf * (alpha * x - F_curr[:, k])
                    # Cの更新: C[:, k] -= epsr * (beta * (V + mu * rO) + C[:, k] + mu * Id[:, k])
                    C_curr[:, k] -= epsr * (beta * (V + mu * rO) + C_curr[:, k] + mu * Id[:, k])
            else:
                O = 0
                
            rO = (1 - leak * dt) * rO
            if O == 1:
                rO[k] += 1.0

            # --- Readout Learning (常に学習) ---
            y_est_vec = np.dot(W_out, rO) + b_out
            pred_idx = np.argmax(y_est_vec)
            
            # 誤差逆伝播
            error_vec = target_vec - y_est_vec
            
            # 重みとバイアスの更新
            W_out += lr_readout * np.outer(error_vec, rO)
            b_out += lr_readout * error_vec
            
            if pred_idx == label:
                img_correct_counts += 1
        
        is_correct = 1 if (img_correct_counts / Duration) > 0.5 else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0) 
        acc_history.append(np.mean(acc_buffer))

    print(f"\nPhase 2 Completed. Final Accuracy: {acc_history[-1]:.4f}")
    
    return acc_history, spike_times, spike_neurons


if __name__ == "__main__":
    # 保存ディレクトリの設定
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_MNIST_Retrain_Exp')
    save_dir = os.path.join('results', run_id)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    print(f"Results will be saved to: {save_dir}")

    # データ読み込み
    X_train, y_train, X_test, y_test = bu.load_mnist_data()

    # パラメータ設定
    Nx = 784        
    Nneuron = 1000   
    leak = 50
    dt = 0.001
    Thresh = 0.5
    Gain = 30.0     
    
    # 学習ルール用パラメータ
    epsr = 0.0001
    epsf = 0.00001
    alpha = 0.18    
    beta = 1 / 0.9
    mu = 0.02 / 0.9

    # 事前学習済み重みのロード (なければ学習)
    weight_file = f'mnist_weights_N{Nneuron}_G{int(Gain)}_Brendelonly.npz'
    
    if os.path.exists(weight_file):
        print(f"Loading weights from {weight_file}...")
        try:
            data = np.load(weight_file)
            F_init = data['F']
            C_init = data['C']
        except:
            print("Weight file error. Retraining initial structure...")
            F_init, C_init = bu.train_snn_structure_mnist(
                X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain
            )
            np.savez(weight_file, F=F_init, C=C_init)
    else:
        print("Pre-training initial structure...")
        F_init, C_init = bu.train_snn_structure_mnist(
            X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain
        )
        np.savez(weight_file, F=F_init, C=C_init)

    # 実験設定
    NumSamples = 2000
    RotationStart = 250
    RetrainStart = 500

    # === 学習実行 ===
    
    # 1. Retrain Model (コンセプトドリフト後に再学習するモデル)
    acc, spk_t, spk_i = train_readout_mnist_Rotation_Retrain(
        F_init, C_init, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain,
        epsf, epsr, alpha, beta, mu,
        label_suffix="Retrain",
        NumSamples=NumSamples,
        rotation_start_idx=RotationStart,
        retrain_start_idx=RetrainStart
    )

    # 2. Non-retrain Model (再学習なし・比較用)
    
    acc_trained, spk_t_trained, spk_i_trained = bu.train_readout_mnist_Rotation(
        F_init, C_init, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain, 
        label_suffix="NoRetrain", 
        NumSamples=NumSamples
    )
    
    # --- プロット1: 正解率 (Accuracy) の比較 ---
    plt.figure(figsize=(10, 6))
    
    # Retrainモデル（青）
    plt.plot(acc, label='With Retraining (F & C adapt)', color='blue', linewidth=1.5)
    
    # No-Retrainモデル（オレンジ・点線）
    plt.plot(acc_trained, label='No Retraining (Readout only)', color='red', linestyle='--', linewidth=1.5)
    
    # 垂直線: 回転開始
    plt.axvline(x=RotationStart, color='red', linestyle=':', label=f'Rotation Start ({RotationStart})')
    # 垂直線: 再学習開始
    plt.axvline(x=RetrainStart, color='green', linestyle='-.', label=f'Retraining Start ({RetrainStart})')
    
    plt.title(f'Accuracy Comparison: Retraining vs No-Retraining')
    plt.xlabel('Sample Index')
    plt.ylabel('Accuracy (Moving Avg)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    
    # ファイル保存
    acc_path = os.path.join(save_dir, 'Accuracy_Comparison.png')
    plt.savefig(acc_path)
    print(f"Saved accuracy comparison to {acc_path}")
    plt.close()
    
    # --- プロット2: ラスタープロットの比較 (上下に並べる) ---
    # ラスターは重ねると見づらいため、上下のSubplotで比較します
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    max_time = 500 * 30 * dt 
    display_samples = 1000
    display_time = display_samples * 30 * dt
    
    # 上段: No Retrain
    axes[0].scatter(spk_t_trained, spk_i_trained, s=2, c='orange', marker='|', alpha=0.6)
    axes[0].set_title("Raster Plot: No Retraining")
    axes[0].set_ylabel("Neuron Index")
    axes[0].set_ylim(0, Nneuron)
    axes[0].axvline(x=RotationStart * 30 * dt, color='red', linestyle='--', label='Rotation')
    
    # 下段: With Retrain
    axes[1].scatter(spk_t, spk_i, s=2, c='blue', marker='|', alpha=0.6)
    axes[1].set_title("Raster Plot: With Retraining (F & C adapt)")
    axes[1].set_ylabel("Neuron Index")
    axes[1].set_xlabel("Time (s)") # X軸ラベルは一番下だけ
    axes[1].set_ylim(0, Nneuron)
    axes[1].axvline(x=RotationStart * 30 * dt, color='red', linestyle='--', label='Rotation')
    axes[1].axvline(x=RetrainStart * 30 * dt, color='green', linestyle='-.', label='Retrain Start')
    
    # 共通設定
    plt.xlim(0, display_time)
    
    # ファイル保存
    raster_path = os.path.join(save_dir, 'Raster_Comparison.png')
    plt.tight_layout()
    plt.savefig(raster_path)
    print(f"Saved raster comparison to {raster_path}")
    plt.close()

    print("All tasks completed.")