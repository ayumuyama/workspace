import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import brendel_utils as bu

if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_MNIST_Tuned')
    save_dir = os.path.join('results', run_id)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    X_train, y_train, X_test, y_test = bu.load_mnist_data()

    # パラメータ設定 (チューニング済み)
    Nx = 784        
    Nneuron = 1000   
    leak = 50
    dt = 0.001
    Thresh = 0.5
    
    # 【重要】Gain調整: 20だと弱すぎるので30に。クリップ処理を入れたので爆発しないはず
    Gain = 30.0     
    
    epsr = 0.0001
    epsf = 0.00001
    alpha = 0.18    
    beta = 1 / 0.9
    mu = 0.02 / 0.9

    Nit_structure = 50000
    Duration_structure = 30

    # 重みファイル名
    weight_file = f'mnist_weights_N{Nneuron}_G{int(Gain)}.npz'
    
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
                X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain,
                Nit=Nit_structure,
                Duration=Duration_structure
            )
            np.savez(weight_file, F=F_trained, C=C_trained)
    else:
        F_trained, C_trained = bu.train_snn_structure_mnist(
            X_train, dt, leak, epsr, epsf, alpha, beta, mu, Nneuron, Nx, Thresh, Gain
        )
        np.savez(weight_file, F=F_trained, C=C_trained)

    # 1. Trained Model
    acc_trained, spk_t_trained, spk_i_trained = bu.train_readout_mnist(
        F_trained, C_trained, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain, label_suffix="Trained"
    )
    
    # 2. Random Reservoir
    print("Initializing Random Reservoir...")
    F_rand, C_rand = bu.init_weights(Nx, Nneuron)
    acc_rand, spk_t_rand, spk_i_rand = bu.train_readout_mnist(
        F_rand, C_rand, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain, label_suffix="Random"
    )

    # --- プロット1: 正解率 ---
    plt.figure(figsize=(10, 6))
    plt.plot(acc_trained, label='Trained SNN', color='blue')
    plt.plot(acc_rand, label='Random Reservoir', color='orange', linestyle='--')
    plt.title(f'MNIST Accuracy (Full History)')
    plt.xlabel('Training Samples'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True); plt.ylim(0, 1.0)
    plt.savefig(os.path.join(save_dir, '1_Accuracy.png'))
    
    # --- プロット2: ラスタープロット (全期間) ---
    print("Plotting full raster plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 点を小さく(s=0.05)、薄く(alpha=0.3)して密度を表現
    ax1.scatter(spk_t_trained, spk_i_trained, s=0.5, c='blue', marker='.', alpha=1.0)
    ax1.set_title("Raster Plot: Trained SNN (Full Learning Process)")
    ax1.set_ylabel("Neuron Index"); ax1.set_ylim(0, Nneuron)

    ax2.scatter(spk_t_rand, spk_i_rand, s=0.5, c='blue', marker='.', alpha=1.0)
    ax2.set_title("Raster Plot: Random Reservoir (Full Learning Process)")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Neuron Index"); ax2.set_ylim(0, Nneuron)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_Full_Raster.png'))

    # --- プロット3: ラスタープロット (学習後期 拡大版) ---
    print("Plotting zoomed raster plots...")
    fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 最後の5秒間（約100サンプル分）を表示
    max_time = 5000 * 50 * dt # 250秒
    zoom_start = max_time - 5.0
    
    ax3.scatter(spk_t_trained, spk_i_trained, s=2, c='blue', marker='|')
    ax3.set_title("Raster Plot: Trained SNN (Last 100 Samples)")
    ax3.set_ylabel("Neuron Index"); ax3.set_ylim(0, Nneuron)
    ax3.set_xlim(zoom_start, max_time)
    
    ax4.scatter(spk_t_rand, spk_i_rand, s=2, c='blue', marker='|')
    ax4.set_title("Raster Plot: Random Reservoir (Last 100 Samples)")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Neuron Index"); ax4.set_ylim(0, Nneuron)
    ax4.set_xlim(zoom_start, max_time)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_Zoomed_Raster.png'))

    print("Done.")