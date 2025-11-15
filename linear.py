import numpy as np
import pickle
import torch
import os

#파라미터 지정
num_classes = 10    # CIFAR-10
n_train = 5000      # 처음엔 작게 시작 (속도↑)
n_val   = 1000
n_test  = 1000
lr = 1e-2          # learning rate
reg = 1e-4          # L2 정규화
batch_size = 128
epochs = 10

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
base = r"C:\Users\haggi\PycharmProjects\cn231_2week\.venv\cifar-10-batches-py"
def SoftMax_(W,b,X,Y,reg):
    #W: (C,D), b: (C,), X: (B,D), y: (B,)
    #return: loss(float), dW(C,D), db(C,)
    B = X.shape[0]#X의 크기
    score = X @ W.T + b
    #sofrMax 계싼
    m = score.max(axis=1, keepdims=True)  # (B,1)
    st = score - m
    lse = np.log(np.exp(st).sum(axis=1, keepdims=True) + 1e-12)  # (B,1)
    log_probs = st - lse  # (B,C)
    SoftMax_score = np.exp(log_probs)

    #loss 계산
    correct = SoftMax_score[np.arange(B),Y]
    data_loss = -np.log(correct+1e-12).mean()
    reg_loss = 0.6*reg*(W*W).sum()
    loss = data_loss + reg_loss

    #정답 클래스일 땐 확률 - 1, 나머진 그냥 확률. 그리고 전부 1/B로 나눔(그래디언트)
    dscores = SoftMax_score
    dscores[np.arange(B),Y] = -1
    dscores /= B

    dW = dscores.T @ X + reg * W
    db = dscores.sum(axis=0)
    return dW,db,loss

def accuracy(W, b, X, y):
    scores = X @ W.T + b
    pred = scores.argmax(axis=1)
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

#초기 W와 바이어스 B 생성
N, D = X_tr.shape
C = num_classes
rng = np.random.default_rng(0)
W = 0.001 * rng.standard_normal(size = (C,D)).astype(np.float32)
B = np.zeros((C,),dtype=np.float32)

num_iters = int(np.ceil(N*batch_size /epochs))
idx = np.arange(N)

for it in range(num_iters):
    sample = rng.choice(idx,size = batch_size,replace = False)
    Xsample = X_tr[sample]
    Yssample = Y_tr[sample]
    dW,db,loss = SoftMax_(W,B,Xsample,Yssample,reg)

    #업데이트
    W -= lr * dW
    B -= lr * db

    if (it + 1) % 100 == 0:
        train_acc = accuracy(W, B, X_tr[:1000], Y_tr[:1000])  # 빠른 추정
        val_acc = accuracy(W, B, X_val, Y_val)
        print(f"[{it + 1}/{num_iters}] loss={loss:.3f}  train@1k={train_acc:.3f}  val={val_acc:.3f}")
        print("val acc:", accuracy(W, B, X_val, Y_val))
        print("test acc:", accuracy(W, B, X_test, Y_test))