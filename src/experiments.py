import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys

# パス設定と自作モジュールの読み込み
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import brendel_random_mnist_utils as bu

if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_expriments')
    save_dir = os.path.join('results', run_id)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # データのロード
    X_train, y_train, X_test, y_test = bu.load_mnist_data()

    # パラメータ設定
    Nx = 784        
    Nneuron = 1000   
    leak = 50
    dt = 0.001
    Thresh = 0.5
    Gain = 30.0     
    
    # 学習パラメータ
    epsr = 0.0006
    epsf = 0.00006
    alpha = 0.18    
    beta = 1 / 0.9
    mu = 0.02 / 0.9

    lr_readout = 0.008

    # 重みファイル名
    weight_file = f'mnist_weights_N{Nneuron}_G{int(Gain)}_Brendelonly.npz'
    
    # 事前学習済みの重みをロード（なければ作成）
    if os.path.exists(weight_file):
        print(f"Loading weights from {weight_file}...")
        try:
            data = np.load(weight_file)
            F_trained = data['F']
            C_trained = data['C']
            if np.isnan(F_trained).any(): raise ValueError
        except:
            print("Weight file corrupted. Retraining...")
            F_trained, C_trained = bu.train_snn_structure_mnist(
                X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain
            )
            np.savez(weight_file, F=F_trained, C=C_trained)
    else:
        F_trained, C_trained = bu.train_snn_structure_mnist(
            X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain
        )
        np.savez(weight_file, F=F_trained, C=C_trained)

    # === シナリオ1: 固定重み (Fixed) ===
    print("\n=== Scenario 1: Fixed Weights (Rotation at 1000) ===")
    acc_fixed, spk_t_fixed, spk_i_fixed = bu.train_readout_mnist_Rotation(
        F_trained, C_trained, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain, 
        label_suffix="Fixed", lr_readout=lr_readout
    )

    # === シナリオ2: 再学習あり (Retrain) ===
    print("\n=== Scenario 2: Retrain from 900 (Rotation at 1000) ===")
    acc_retrain, spk_t_retrain, spk_i_retrain = bu.train_readout_mnist_Rotation_Retrain(
        F_trained, C_trained, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain,
        epsf, epsr, alpha, beta, mu,
        label_suffix="Retrain", lr_readout=lr_readout, retrain_start=900
    )
    
    # --- プロット1: 正解率の比較 (重ねて表示) ---
    plt.figure(figsize=(10, 6))
    plt.plot(acc_fixed, label='Fixed SNN', color='blue', linewidth=1.5)
    plt.plot(acc_retrain, label='Retrained SNN (from 900)', color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=1000, color='green', linestyle=':', label='Rotation Start')
    plt.axvline(x=900, color='orange', linestyle=':', label='Retrain Start')
    plt.title(f'Accuracy Comparison: Fixed vs Retrained')
    plt.xlabel('Training Samples'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True); plt.ylim(0, 1.0)
    plt.savefig(os.path.join(save_dir, '1_Accuracy_Comparison.png'))
    
    # --- プロット2: ラスタープロット (上下に配置) ---
    print("Plotting combined raster plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    max_time = 7000 * 30 * dt # 7000 samples * 30 steps * dt

    # 上段: Fixed
    ax1.scatter(spk_t_fixed, spk_i_fixed, s=2, c='blue', marker='|')
    ax1.axvline(x=1000 * 30 * dt, color='black', linestyle='--', label='Rotation')
    ax1.set_title("Raster: Fixed SNN")
    ax1.set_ylabel("Neuron Index")
    ax1.set_ylim(0, Nneuron)
    
    # 下段: Retrain
    ax2.scatter(spk_t_retrain, spk_i_retrain, s=2, c='red', marker='|')
    ax2.axvline(x=1000 * 30 * dt, color='black', linestyle='--', label='Rotation')
    ax2.axvline(x=900 * 30 * dt, color='black', linestyle='--', label='Retrain')
    ax2.set_title("Raster: Retrained SNN")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, max_time)
    ax2.set_ylabel("Neuron Index")
    ax2.set_ylim(0, Nneuron)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_Raster_Comparison.png'))

    # --- プロット3: 再学習モデルの拡大ラスタープロット (25s〜50s) ---
    print("Plotting zoomed raster plot for retrained model (25s-50s)...")
    plt.figure(figsize=(15, 6))

    # 描画範囲の設定
    t_start = 25.0
    t_end = 50.0

    # データをNumPy配列に変換（フィルタリング操作のため）
    spk_t_retrain_np = np.array(spk_t_retrain)
    spk_i_retrain_np = np.array(spk_i_retrain)

    # 指定した時間範囲内のスパイクのみを抽出するマスクを作成
    mask = (spk_t_retrain_np >= t_start) & (spk_t_retrain_np <= t_end)

    # フィルタリングされたデータで散布図を描画
    # マーカーサイズ(s)を少し大きくして見やすくしています
    plt.scatter(spk_t_retrain_np[mask], spk_i_retrain_np[mask], s=5, c='red', marker='|')

    # イベント発生時刻の計算
    rotation_time = 1000 * 30 * dt  # 30.0s
    retrain_time = 900 * 30 * dt   # 60.0s

    # 範囲内にイベントが含まれていれば垂直線を描画
    if t_start <= rotation_time <= t_end:
        plt.axvline(x=rotation_time, color='black', linestyle='--', linewidth=2, label='Rotation Start (1000 samples)')
    # retrain_time (60s) は今回の範囲 (25-50s) には含まれませんが、汎用的な記述として残します
    if t_start <= retrain_time <= t_end:
        plt.axvline(x=retrain_time, color='black', linestyle='--', linewidth=2, label='Retrain Start (2000 samples)')

    # グラフの装飾
    plt.title(f"Zoomed Raster Plot: Retrained SNN ({t_start}s - {t_end}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index")
    plt.xlim(t_start, t_end)
    plt.ylim(0, Nneuron)
    plt.legend(loc='upper right')
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')

    plt.tight_layout()
    # ファイル名を指定して保存
    plt.savefig(os.path.join(save_dir, '3_Raster_Retrain_Zoomed_25s-50s.png'))

    print(f"Done. Results saved in {save_dir}")