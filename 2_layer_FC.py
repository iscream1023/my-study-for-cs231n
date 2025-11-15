import numpy as np
import pickle
import torch
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

#파라미터 지정
num_classes = 10    # CIFAR-10
n_train = 5000      # 처음엔 작게 시작 (속도↑)
n_val   = 1000
n_test  = 1000
lr = 1e-2           # learning rate
reg = 1e-4          # L2 정규화
batch_size = 256
epochs = 20
base = r"C:\Users\haggi\PycharmProjects\cn231_2week\.venv\cifar-10-batches-py"
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def layer(W1, B1,W2,B2, X, y,reg):
    #print("X", X.shape, X.dtype, np.nanmin(X), np.nanmax(X))
    #print("W1", W1.shape, np.max(np.abs(W1)))
    #print("W2", W2.shape, np.max(np.abs(W2)))
    #print("b1", B1.shape, "b2", B2.shape)
    #print("y range:", int(y.min()), int(y.max()))

    B = X.shape[0]#X의 크기
    hidden_lin = X @ W1.T + B1  # (B,H)  pre-activation 저장
    hidden = np.maximum(0, hidden_lin)
    scores = hidden @ W2.T + B2

    #sofrMax 계싼
    st = scores - scores.max(axis=1, keepdims=True)  # (B,C)
    exp = np.exp(st)
    SoftMax_score = exp / exp.sum(axis=1, keepdims=True)
    #loss 계산
    correct = SoftMax_score[np.arange(B), y]
    data_loss = -np.log(correct + 1e-12).mean()
    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    loss = data_loss + reg_loss

    #정답 클래스일 땐 확률 - 1, 나머진 그냥 확률. 그리고 전부 1/B로 나눔(그래디언트)
    dscores = SoftMax_score
    dscores[np.arange(B),y] -= 1
    dscores /= B

    dW2 = dscores.T @ hidden + reg * W2
    db2 = dscores.sum(axis=0)
    dhidden = dscores @ W2
    dhidden[hidden_lin <= 0] = 0#ReLu

    dW1 = dhidden.T @ X + reg * W1
    db1 = dhidden.sum(axis=0)
    return dW1,db1,dW2,db2,loss

def accuracy(W1,B1,W2,B2, X, y):
    Hidden = np.maximum(0, X @ W1.T + B1)
    score = Hidden @ W2.T + B2
    pred = score.argmax(axis=1)
    return (pred == y).mean()

#데이터 받아오기
X_train_list = []
y_train_list = []
for i in range(1, 6):#5개의 batch파일을 모드 로드->2차원
    batch = unpickle(os.path.join(base, f"data_batch_{i}"))
    X_train_list.append(batch[b'data'])     # (10000,3072)
    y_train_list.extend(batch[b'labels'])   # list of 10000
X_train = np.concatenate(X_train_list,0)#2차원을 1차원으로 나열
Y_train = np.array(y_train_list, dtype=np.int64)

bte = unpickle(os.path.join(base, f"test_batch"))
X_test = bte[b'data']
Y_test = np.array(bte[b'labels'],dtype=np.int64)

#데이터 전처리 - 각 데이터를 트레인/벨리데이션/테스트 용으로 구분
X_tr = X_train[:n_train].astype(np.float32)
Y_tr = Y_train[:n_train]
X_val = X_train[n_train:n_val+n_train].astype(np.float32)
Y_val = Y_train[n_train:n_val+n_train]
X_test = X_test[:n_test].astype(np.float32)
Y_test = Y_test[:n_test]

X_tr = X_tr.astype(np.float32) / 255.0
X_val = X_val.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

#표준화
mean = X_tr.mean(axis=0, keepdims=True)
sig = X_tr.std(axis=0, keepdims=True)+ 1e-8
X_tr = (X_tr - mean)/sig
X_val = (X_val - mean)/sig
X_test = (X_test - mean)/sig
#데이터 차원 확인
print("shapes:", X_tr.shape, X_val.shape, X_test.shape)
#그래프 그리기
plt.ion()                         # 인터랙티브 ON
fig, ax1 = plt.subplots()
(train_line,) = ax1.plot([], [], label="train")
(val_line,)   = ax1.plot([], [], label="val")
ax1.set_ylim(0, 1.0); ax1.set_xlim(0, 1)
ax1.set_xlabel("epoch"); ax1.set_ylabel("Acc"); ax1.legend()
fig.canvas.draw(); fig.canvas.flush_events()
ax2 = ax1.twinx()
(loss_line,)  = ax2.plot([], label="loss", color="purple", linestyle="--")
ax2.set_ylabel("Loss", color="purple")
ax2.legend(loc="upper right")

#초기 W와 바이어스 B 생성
rng = np.random.default_rng(0)
N, D = X_tr.shape
C = num_classes
W1 = 0.001 * rng.standard_normal(size = (100,D)).astype(np.float32)
B1 = np.zeros((100,),dtype=np.float32)
W2 = 0.001 * rng.standard_normal(size = (C,100)).astype(np.float32)
B2 = np.zeros((C,),dtype=np.float32)

num_iters = int(np.ceil(N*batch_size /epochs))
idx = np.arange(N)
train_hist, val_hist, loss_hist = [], [], []

for it in range(num_iters):
    sample = rng.choice(idx,size = batch_size,replace = False)
    Xsample = X_tr[sample]
    Ysample = Y_tr[sample]
    #dW,db,loss = SoftMax_(W,B,Xsample,Yssample,reg)
    dW1, db1, dW2, db2, loss = layer(W1,B1,W2,B2, Xsample, Ysample,reg)
    #업데이트
    W1 -= lr * dW1
    B1 -= lr * db1
    W2 -= lr * dW2
    B2 -= lr * db2
    if (it + 1) % 100 == 0:
        train_acc = accuracy(W1, B1, W2, B2, X_tr[:5000], Y_tr[:5000])
        val_acc = accuracy(W1, B1, W2, B2, X_val, Y_val)
        print(f"[{it + 1}/{num_iters}] loss={loss:.3f}  train@1k={train_acc:.3f}  val={val_acc:.3f}")

        train_hist.append(train_acc)
        val_hist.append(val_acc)
        loss_hist.append(loss)

        xs = list(range(1, len(train_hist) + 1))
        train_line.set_data(xs, train_hist)
        val_line.set_data(xs, val_hist)
        loss_line.set_data(xs, loss_hist)
        ax1.set_xlim(1, max(2, len(xs)))  # x축 자동 확장
        ax1.relim();
        ax1.autoscale_view(scalex=False, scaley=True)
        ax2.relim();
        ax2.autoscale_view(scalex=False, scaley=True)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

test_acc = accuracy(W1, B1, W2, B2, X_te, Y_te)
print("Final test accuracy:", test_acc)
plt.ioff()
plt.show()
