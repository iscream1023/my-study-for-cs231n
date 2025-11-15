import numpy as np
import pickle
import torch
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class KNN:
    def __init__(self):
        pass

    def train(self, x, y):
        self.Xtr = x
        self.ytr = y
        print("Xtr shape:", self.Xtr.shape)
        print("ytr shape:", self.ytr.shape)

    def predict_L1(self, X,k=1):
        print("starting Predici...")
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            # L1 distance (맨해튼 거리)
            distance = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            knn_index = np.argmin(distance)[:k]
            knn_labels = self.ytr[knn_index]
            counts = np.bincount(knn_labels, minlength=10)
            Ypred[i] = self.ytr[counts]
        return Ypred

    def predict_L2(self,X,K=1):
        print("starting Predici...")
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            # L2 distance (유클리드 거리)
            distance = np.sum((self.Xtr - X[i, :]) ** 2, axis=1)
            knn_index = np.argmin(distance)[:k]
            knn_labels = self.ytr[knn_index]
            counts = np.bincount(knn_labels, minlength=10)
            Ypred[i] = self.ytr[counts]

        return Ypred

base = r"C:\Users\haggi\PycharmProjects\cn231_2week\.venv\cifar-10-batches-py"
#훈련 셋 로드
X_train_list = []
y_train_list = []
#for i in range(1, 6):
batch = unpickle(os.path.join(base, f"data_batch_{1}"))
X_train_list.append(batch[b'data'])     # (10000,3072)
y_train_list.extend(batch[b'labels'])   # list of 10000

X_train = np.concatenate(X_train_list, axis=0)  # (50000,3072)
y_train = np.array(y_train_list, dtype=np.int64)
#테스트 셋 로드
test_batch = unpickle(os.path.join(base, "test_batch"))
X_test = test_batch[b'data']                         # (10000,3072)
y_test = np.array(test_batch[b'labels'], dtype=np.int64)
# 실행
knn = KNN()
knn.train(X_train, y_train)
Knum = 5 #KNN에서 고려할 주변 데이터 개수
y_pred_L1 = knn.predict_L1(X_test[:200],k=Knum)
y_pred_L2 = knn.predict_L2(X_test[:200],k=Knum)

acc_L1 = (y_pred_L1 == y_test[:len(y_pred)]).mean()
acc_L2 = (y_pred_L2 == y_test[:len(y_pred)]).mean()
print(f"Accuracy on {X_test} test using L1 train: {acc_L1:.3f}")
print(f"Accuracy on {X_test} test using L2 train: {acc_L2:.3f}")
