import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import brendel_utils as bu

class ForceESN:
    def __init__(self, n_inputs, n_neurons, n_outputs, dt=0.001, spectral_radius=1.2):
        self.N = n_neurons
        self.dt = dt
        self.alpha = 100 * dt  
        
        np.random.seed(42)
        self.Win = np.random.randn(n_neurons, n_inputs) * 1.0
        W = np.random.randn(n_neurons, n_neurons)
        eigenvalues = np.linalg.eigvals(W)
        self.W = W * (spectral_radius / np.max(np.abs(eigenvalues)))
        self.Wout = np.zeros((n_outputs, n_neurons))
        
        self.P = np.eye(n_neurons) / 1.0
        self.r = np.zeros(n_neurons)
        self.x = np.zeros(n_neurons)

    def step(self, inp, target, training=True):
        self.x = (1 - self.alpha) * self.x + self.alpha * (np.dot(self.Win, inp) + np.dot(self.W, self.r))
        self.r = np.tanh(self.x)
        
        z = np.dot(self.Wout, self.r)
        
        if training:
            e = z - target
            Pr = np.dot(self.P, self.r)
            rPr = np.dot(self.r, Pr)
            c = 1.0 / (1.0 + rPr)
            k = Pr * c
            self.Wout -= np.outer(e, k)
            self.P -= np.outer(k, Pr)
            
        return z

if __name__ == "__main__":
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_RateBased_Efficient')
    save_dir = os.path.join('results', run_id)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = bu.load_mnist_data()
    
    # === 設定 ===
    Nx = 784
    Nneuron = 1000
    Nclasses = 10
    
    NumSamples = 7000   
    Duration = 30       
    dt = 0.001          
    ChangePoint = 1000   # 7.5秒時点で回転
    
    TotalTime = NumSamples * Duration * dt # 60.0s
    
    # === 【重要】記録・表示したい範囲をここで定義 ===
    # この範囲外のデータはメモリに保存しません
    view_start_time = 25.0
    view_end_time = 50.0
    
    print(f"Simulation Settings: Samples={NumSamples}, TotalTime={TotalTime:.1f}s")
    print(f"Recording Activity ONLY for: {view_start_time}s - {view_end_time}s")
    
    esn = ForceESN(Nx, Nneuron, Nclasses, dt=dt)
    
    acc_history = []
    acc_buffer = []
    
    # ダウンサンプリング設定（さらにメモリ節約）
    downsample_rate = 5 
    activity_log = []
    
    print("Starting Training...")
    global_step = 0
    
    for i in range(NumSamples):
        if i % 100 == 0:
            print(f'\r  Sample {i}/{NumSamples} (Time: {i*Duration*dt:.1f}s)', end='')
            
        raw_img = X_train[i % len(X_train)]
        if i >= ChangePoint:
            raw_img = np.rot90(raw_img.reshape(28, 28), k=1).flatten()
            
        label = y_train[i % len(y_train)]
        target = np.zeros(Nclasses)
        target[label] = 1.0
        
        img_correct = 0
        
        for t in range(Duration):
            # 現在時刻の計算
            current_time = (i * Duration + t) * dt
            
            output = esn.step(raw_img, target, training=True)
            
            if np.argmax(output) == label:
                img_correct += 1
            
            # === 修正ポイント: 指定範囲内のときだけ記録 ===
            if view_start_time <= current_time <= view_end_time:
                # 範囲内のみカウントしてダウンサンプリング
                global_step += 1
                if global_step % downsample_rate == 0:
                    activity_log.append(esn.r.copy())
        
        is_correct = 1 if (img_correct / Duration) > 0.5 else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > 200: acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))

    print(f"\nCompleted. Final Accuracy: {acc_history[-1]:.4f}")

    # === プロット1: 正解率 (これは全期間表示) ===
    time_axis_samples = np.arange(NumSamples) * Duration * dt 
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis_samples, acc_history, label='Rate-based ESN (FORCE)', color='green')
    plt.axvline(x=ChangePoint * Duration * dt, color='red', linestyle='--', label='Input Change (7.5s)')
    
    # 記録した範囲を黄色くハイライト
    plt.axvspan(view_start_time, view_end_time, color='yellow', alpha=0.2, label='Raster Plot Range')
    
    plt.title(f'Accuracy (Full Range: 0-{TotalTime:.0f}s)')
    plt.xlabel('Time (s)'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True); plt.ylim(0, 1.0)
    plt.savefig(os.path.join(save_dir, '1_Accuracy_Full.png'))
    
    # === プロット2: ニューロン活動ヒートマップ (記録した範囲のみ) ===
    print(f"Generating activity map ({view_start_time}-{view_end_time}s)...")
    
    if len(activity_log) == 0:
        print("Warning: No activity data recorded. Check the time range settings.")
    else:
        plt.figure(figsize=(15, 6))
        
        # 記録されたデータはすでに指定範囲内のみなので、そのまま使う
        activity_data = np.array(activity_log).T 
        
        # extentで軸の数値を指定範囲に合わせる
        extent = [view_start_time, view_end_time, 0, Nneuron]
        
        plt.imshow(activity_data, aspect='auto', cmap='bwr', interpolation='nearest', 
                   vmin=-1, vmax=1, extent=extent)
        
        plt.colorbar(label='Activation (tanh)')
        
        if view_start_time <= (ChangePoint * Duration * dt) <= view_end_time:
            plt.axvline(x=ChangePoint * Duration * dt, color='black', linestyle='--', linewidth=1.5, label='Input Change')
        
        plt.title(f"Reservoir Activity (Efficiently Recorded: {view_start_time}-{view_end_time}s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Neuron Index")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '2_Activity_Efficient.png'))
    
    print(f"Results saved to {save_dir}")