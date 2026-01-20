import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys

# パス設定と自作モジュールの読み込み
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import brendel_random_mnist_utils as bu

if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_MNIST_Rotation_Compare')
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
    NumSample = 10000
    retrain_start = 0

    # 重みファイル名
    weight_file = f'mnist_weights_N{Nneuron}_G{int(Gain)}_epsr{epsr}_epsf{epsf}_mu{mu}_Brendelonly.npz'
    
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
    acc_fixed, spk_t_fixed, spk_i_fixed = bu.train_readout_mnist_Rotation_return(
        F_trained, C_trained, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain, 
        label_suffix="Fixed", NumSamples=NumSample, lr_readout=lr_readout
    )

    # === シナリオ2: 再学習あり (Retrain) ===
    print("\n=== Scenario 2: Retrain (Rotation at 1000) ===")
    acc_retrain, spk_t_retrain, spk_i_retrain = bu.train_readout_mnist_Rotation_Retrain_return(
        F_trained, C_trained, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain,
        epsf, epsr, alpha, beta, mu,
        label_suffix="Retrain", retrain_start=retrain_start, NumSamples=NumSample, lr_readout=lr_readout
    )
    
    # --- プロット1: 正解率の比較 (重ねて表示) ---
    plt.figure(figsize=(10, 6))
    plt.plot(acc_fixed, label='Fixed SNN', color='blue', linewidth=1.5)
    plt.plot(acc_retrain, label='Retrained SNN', color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=1000, color='red', linestyle=':', label='Rotation Start') # ラベルと色を変更
    plt.axvline(x=5000, color='green', linestyle=':', label='Return to Normal & Retrain Stop') # ラベルと色を変更
    plt.title(f'Accuracy Comparison: Fixed vs Retrained')
    plt.xlabel('Training Samples'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True); plt.ylim(0, 1.0)
    plt.savefig(os.path.join(save_dir, '1_Accuracy_Comparison_return.png'))
    
    # --- プロット2: ラスタープロット (上下に配置) ---
    print("Plotting combined raster plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    max_time = 7000 * 30 * dt # 1500 samples * 30 steps * dt

    # 上段: Fixed
    zoom_start = 900
    zoom_end = 5500
    Duration = 30
    ax1.set_xlim(zoom_start * Duration * dt, zoom_end * Duration * dt)
    ax1.scatter(spk_t_fixed, spk_i_fixed, s=2, c='blue', marker='|')
    ax1.axvline(x=1000 * 30 * dt, color='red', linestyle='--', label='Rotation start')
    ax1.axvline(x=5000 * 30 * dt, color='green', linestyle='--', label='Rotation stop')
    ax1.set_title("Raster: Fixed SNN")
    ax1.set_ylabel("Neuron Index")
    ax1.set_ylim(0, Nneuron)
    
    # 下段: Retrain
    ax2.set_xlim(zoom_start * Duration * dt, zoom_end * Duration * dt)
    ax2.scatter(spk_t_retrain, spk_i_retrain, s=2, c='red', marker='|')
    ax2.axvline(x=1000 * 30 * dt, color='red', linestyle='--', label='Rotation start')
    ax2.axvline(x=5000 * 30 * dt, color='green', linestyle='--', label='Rotation stop&Retrain Stop')
    ax2.set_title("Raster: Retrained SNN")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Neuron Index")
    ax2.set_ylim(0, Nneuron)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_Raster_Comparison_return.png'))

    print(f"Done. Results saved in {save_dir}")