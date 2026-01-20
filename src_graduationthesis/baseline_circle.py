import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import deque

# ==========================================
# 設定 (SNNのコードと合わせる)
# ==========================================
CSV_PATH = 'data/BC-001.csv'
INPUT_COLS = [0, 1]
LABEL_COL = 2

# 移動平均の窓幅
WINDOW_SIZE = 50 

# スライディングウィンドウ再学習の設定
RETRAIN_WINDOW = 500  # 直近何個のデータを覚えているか
RETRAIN_INTERVAL = 10 # 何ステップごとに再学習するか (毎回やると重いため)

# ==========================================
# データ読み込み (既存コード流用)
# ==========================================
def load_raw_data(csv_path, input_cols, label_col):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Creating dummy XOR data...")
        N_dummy = 2000
        X = np.random.randn(N_dummy, 2)
        Y = []
        for i, x in enumerate(X):
            # 簡易ドリフト: 前半と後半でルール反転
            if i < N_dummy // 2:
                Y.append(1 if x[0]*x[1] < 0 else 0)
            else:
                Y.append(1 if x[0]*x[1] > 0 else 0)
        Y = np.array(Y)
        df = pd.DataFrame(np.column_stack([X, Y]))
    else:
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, header=0)

    raw_inputs = df.iloc[:, input_cols].values
    raw_labels = df.iloc[:, label_col].values
    return raw_inputs, raw_labels

# ==========================================
# ベースライン1: 逐次学習 (Online SGD / MLP)
# scikit-learnの partial_fit を使用
# ==========================================
def run_online_learning(model, inputs, labels, classes, name="Online Model"):
    acc_history = []
    acc_buffer = [] 
    
    print(f"Running {name}...")
    
    for i, (x_np, y) in enumerate(zip(inputs, labels)):
        x = x_np.reshape(1, -1) # sklearnは2次元配列を要求する
        
        # 1. 予測 (未学習の状態だとランダムあるいはエラーになるのでtry-catchか初期化が必要)
        try:
            pred = model.predict(x)[0]
        except:
            # まだ一度もfitしていない場合の初期予測（適当に0とする）
            pred = 0 
        
        # 2. 学習 (1サンプルずつ更新)
        model.partial_fit(x, [y], classes=classes)
        
        # 3. 精度記録
        is_correct = 1 if pred == y else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > WINDOW_SIZE:
            acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))
        
    return acc_history

# ==========================================
# ベースライン2: スライディングウィンドウ再学習
# ドリフト対応決定木の代わり (直近データだけで毎回作り直す)
# ==========================================
def run_sliding_window(inputs, labels, window_size, interval, name="Sliding Window Tree"):
    acc_history = []
    acc_buffer = []
    
    # データ保持用バッファ
    data_buffer = deque(maxlen=window_size)
    label_buffer = deque(maxlen=window_size)
    
    # 内部モデル (決定木など軽量なものが良い)
    model = DecisionTreeClassifier(max_depth=5)
    is_fitted = False
    
    print(f"Running {name}...")

    for i, (x_np, y) in enumerate(zip(inputs, labels)):
        x = x_np.reshape(1, -1)
        
        # 1. 予測
        if is_fitted:
            pred = model.predict(x)[0]
        else:
            pred = 0 # 初期値
            
        # 2. データ蓄積
        data_buffer.append(x_np)
        label_buffer.append(y)
        
        # 3. 再学習 (インターバルごとに実行)
        if i % interval == 0 and len(data_buffer) >= 50:
            # バッファ内のデータでモデルを全学習
            X_batch = np.array(data_buffer)
            Y_batch = np.array(label_buffer)
            # クラスが1種類しかないとエラーになるのでチェック
            if len(np.unique(Y_batch)) > 1:
                model.fit(X_batch, Y_batch)
                is_fitted = True
        
        # 4. 精度記録
        is_correct = 1 if pred == y else 0
        acc_buffer.append(is_correct)
        if len(acc_buffer) > WINDOW_SIZE:
            acc_buffer.pop(0)
        acc_history.append(np.mean(acc_buffer))
        
    return acc_history

# ==========================================
# メイン実行
# ==========================================
if __name__ == "__main__":
    raw_inputs, raw_labels = load_raw_data(CSV_PATH, INPUT_COLS, LABEL_COL)
    all_classes = np.unique(raw_labels)

    # --- Model A: Online Logistic Regression (SGD) ---
    # SNNのLinear Readout部分と比較対象。単純な線形分離。
    sgd_model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=42)
    acc_sgd = run_online_learning(sgd_model, raw_inputs, raw_labels, all_classes, "Online Logistic Regression")

    # --- Model B: Online MLP (Neural Network) ---
    # SNN全体との比較対象。バックプロパゲーションで逐次学習するNN。
    # hidden_layer_sizes=(50,) はあなたのSNNのニューロン数に合わせています
    mlp_model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='sgd', 
                              learning_rate='constant', learning_rate_init=0.01, random_state=42)
    acc_mlp = run_online_learning(mlp_model, raw_inputs, raw_labels, all_classes, "Online MLP (ANN)")

    # --- Model C: Sliding Window Decision Tree ---
    # ドリフト対応モデルの代用。古いデータを忘れることで変化に対応する。
    # Hoeffding Treeの代わりとして機能します。
    acc_tree = run_sliding_window(raw_inputs, raw_labels, RETRAIN_WINDOW, RETRAIN_INTERVAL, "Sliding Window Tree")

    # プロット
    plt.figure(figsize=(12, 6))
    
    # X軸はサンプル数
    plt.plot(acc_tree, label='Sliding Window Tree (Drift Adapted)', linewidth=2, color='green')
    plt.plot(acc_mlp, label='Online MLP (Standard ANN)', linestyle='--', color='orange')
    plt.plot(acc_sgd, label='Online Logistic Reg (Linear)', linestyle=':', color='blue', alpha=0.6)
    
    plt.title('Baseline Models Comparison (Without River)')
    plt.xlabel('Sample Index')
    plt.ylabel('Accuracy (Moving Avg)')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    
    plt.savefig('baseline_sklearn.png')
    print("Saved baseline_sklearn.png")