import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys

# パス設定と自作モジュールの読み込み
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import brendel_random_mnist_utils as bu

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

    # 重みファイル名
    weight_file = f'mnist_weights_N{Nneuron}_G{int(Gain)}_Brendelonly.npz'
    
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

    # 1. Trained Model
    acc_trained, spk_t_trained, spk_i_trained = bu.train_readout_mnist_Rotation(
        F_trained, C_trained, X_train, y_train, Nneuron, Nx, dt, leak, Thresh, Gain, label_suffix="Trained"
    )
    
    # --- プロット1: 正解率 ---
    plt.figure(figsize=(10, 6))
    plt.plot(acc_trained, label='Trained SNN (Precise Timing)', color='blue')
    plt.axvline(x=250, color='red', linestyle='--', label='Input Change')# plt.plot(acc_poisson, label='Poisson Baseline (Rate Matched)', color='green', linestyle='--')
    plt.title(f'Accuracy: Trained Model')
    plt.xlabel('Training Samples'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True); plt.ylim(0, 1.0)
    plt.savefig(os.path.join(save_dir, '1_Accuracy_Comparison.png'))
    
    # --- プロット2: ラスタープロット (拡大版) ---
    print("Plotting raster plot...")
    
    # subplotsを単一のものに変更し、高さ(figsize)を適宜調整しています
    fig, ax = plt.subplots(figsize=(15, 5)) 
    
    max_time = 500 * 30 * dt

    ax.scatter(spk_t_trained, spk_i_trained, s=5, c='blue', marker='|')
    ax.axvline(x=250, color='red', linestyle='--', linewidth=1.5, label='Input Change')
    ax.set_title("Trained SNN: Structured & Sparse")
    ax.set_ylabel("Neuron Index")
    ax.set_ylim(0, Nneuron)
    ax.set_xlim(0, max_time)
    
    # 元々ax2にあったX軸ラベルをこちらに設定
    ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_Raster_Comparison.png'))

    print("Done.")