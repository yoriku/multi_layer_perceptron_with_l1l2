from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight as ccw
import seaborn as sns
import gc
import optuna
import random

class MLP:
    def __init__(self, n_input: int, n_comp: int, 
                 n_output: int, batch_size=256, n_epoch=100, class_weight=None, lambda_l1=0.01, lambda_l2=0.01, seed=42):
        """_summary_

        Args:
            n_input (int): 入力変数の数
            n_comp (int): 中間層の数
            n_output (int): 分類クラス数
            batch_size (int, optional):  Defaults to 256.
            n_epoch (int, optional):  Defaults to 100.
            class_weight (list, optional): y_trainを入れるとSKLEARNにおけるclass_weight=balancedと等価 Defaults to None.
            lambda_l1 (float, optional): L1正則化の強さ. Defaults to 0.01.
            lambda_l2 (float, optional): L2正則化の強さ. Defaults to 0.01.
        """
        torch_fix_seed(seed=seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = Net(n_input, n_comp, n_output).to(self.device)
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.class_weight = class_weight
        self.criterion, self.optimizer = self.set(lambda_l1, lambda_l2)
    
    def set(self, lambda_l1, lambda_l2):
        if(self.class_weight is not None):
            y = torch.from_numpy(self.class_weight).clone()
            w = torch.tensor(ccw('balanced', classes=np.unique(y),
                                 y=y.numpy()), dtype=torch.float).to(self.device)
            criterion = CEL_with_L1L2(weight=w, lambda_l1=lambda_l1, lambda_l2=lambda_l2)
        else:
            criterion = CEL_with_L1L2(weight=None, lambda_l1=lambda_l1, lambda_l2=lambda_l2)
        optimizer = optim.Adam(self.net.parameters(), lr=0.001,
                               weight_decay=1e-6)
        
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

        return criterion, optimizer
    
    def make_dataloader(self, X, y, shuffle=True):
        X = torch.from_numpy(X.astype(np.float32)).clone().to(self.device)
        y = torch.from_numpy(y.astype(np.int64)).clone().to(self.device)

        DATA_SET = TensorDataset(X, y)

        return DataLoader(DATA_SET, batch_size=self.batch_size, shuffle=shuffle)
    
    def fit(self, X, y, verbose=False, val_X=None, val_y=None):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            dict: 各epochにおける更新履歴
        """
        TRAIN = self.make_dataloader(X, y)

        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "l1": [], "l2": []}

        for i in range(self.n_epoch):
            total_loss = 0.0
            total_correct = 0
            self.net.train()
            for x, y in TRAIN:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.net(x)
                loss = self.criterion(y_pred, y, self.net)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(y_pred, 1)
                total_correct += (predicted == y).sum().item()
            accuracy = total_correct / len(TRAIN.dataset)
            # total_f = f1_score(t, p, pos_label=1)
            avg_loss = total_loss / len(TRAIN.dataset)
            history["l1"].append(self.net.state_dict()["l1.weight"].to("cpu"))
            history["l2"].append(self.net.state_dict()["l2.weight"].to("cpu"))
            history["loss"].append(avg_loss)
            history["acc"].append(accuracy)

            if val_X is not None and val_y is not None:
                VAL = self.make_dataloader(val_X, val_y, shuffle=False)
                self.net.eval()

                with torch.no_grad():
                    total_loss = 0.0
                    total_correct = 0
                    for inputs, labels in VAL:
                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, labels, self.net)

                        total_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total_correct += (predicted == labels).sum().item()

                avg_loss = total_loss / len(VAL.dataset)
                accuracy = total_correct / len(VAL.dataset)
                history["val_loss"].append(avg_loss)
                history["val_acc"].append(accuracy)

        if(verbose == True):
            if val_X is not None and val_y is not None:
                plot_graph(history["loss"], history["acc"], self.n_epoch, history["val_loss"], history["val_acc"])
            else:                
                plot_graph(history["loss"], history["acc"], self.n_epoch)
        return history
    
    def predict(self, X, y):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_

        Returns:
            dict: 予測ラベル (pred)と事後確率 (pred_prob)，正解ラベル (true)を返す
        """
        TEST = self.make_dataloader(X, y, shuffle=False)
        results = {"pred": [], "pred_prob": [], "true": []}

        self.net.eval()
        for x, y in TEST:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                y_pred = self.net(x)
                y_pred_prob = nn.functional.softmax(y_pred, dim=1) # softmaxで事後確率に
                y_pred = torch.argmax(y_pred_prob, axis=1).cpu().numpy()
                y_pred_prob = y_pred_prob[:, 1].cpu().numpy().tolist() # high_riskの方の確立を抽出
                y_pred = y_pred.tolist()
            results["pred"] = results["pred"] + y_pred
            results["pred_prob"] = results["pred_prob"] + y_pred_prob
            results["true"] = results["true"] + y.cpu().tolist()

        return results


class AutoMLP(MLP):
    def __init__(self, n_input: int, n_comp: int, 
                 n_output: int, batch_size=256, n_epoch=100, class_weight=None, trial_size=100, seed=42):
        torch_fix_seed(seed=seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_input = n_input
        self.n_comp = n_comp
        self.n_output = n_output
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.class_weight = class_weight
        self.trial_size = trial_size
    
    def objective(self, trial): # validation data!

        lambda_l1 = trial.suggest_float('lambda_l1', 0.0001, 10, log=True)
        lambda_l2 = trial.suggest_float('lambda_l2', 0.0001, 10, log=True)
        n_comp = trial.suggest_int('n_comp', self.n_output, max(int(self.n_input // 2), self.n_output))
        
        self.net = Net(self.n_input, n_comp, self.n_output).to(self.device)

        self.criterion, self.optimizer = self.set(lambda_l1, lambda_l2)
        _ = self.fit(self.tmp_X, self.tmp_y)
        results = self.predict(self.tmp_val_X, self.tmp_val_y)
        mcc = matthews_corrcoef(results["true"], results["pred"])

        del self.net
        torch.cuda.empty_cache()
        gc.collect()
        return 1-mcc
    
    def turning_params(self, X, y, val_X, val_y):
        self.tmp_X = X
        self.tmp_y = y
        self.tmp_val_X = val_X
        self.tmp_val_y = val_y
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=self.trial_size)

        lambda_l1 = study.best_params["lambda_l1"]
        lambda_l2 = study.best_params["lambda_l2"]
        n_comp = study.best_params["n_comp"]
        print(f"lambda_l1: {lambda_l1}, lambda_l2: {lambda_l2}, n_comp {n_comp}")

        self.net = Net(self.n_input, n_comp, self.n_output).to(self.device)
        self.criterion, self.optimizer = self.set(lambda_l1, lambda_l2) # 後はいつも通り



class Net(nn.Module):
    def __init__(self, n_input, n_comp, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input,n_comp)
        self.l2 = nn.Linear(n_comp,n_output) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class CEL_with_L1L2(nn.Module):
    def __init__(self, weight=None, lambda_l1=0.1, lambda_l2=0.1):
        super(CEL_with_L1L2, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.weight = weight

    def forward(self, model_output, target, net):
        loss = nn.functional.cross_entropy(model_output, target, weight=self.weight)
        
        # L1正則化
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in net.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        
        # L2正則化
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in net.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)

        loss = loss + self.lambda_l1 * l1_reg + self.lambda_l2 * l2_reg

        return loss

def plot_graph(values1, values2, rng, val_loss=None, val_acc=None, label1='LOSS', label2='Accuracy'):
    plt.plot(range(rng), values1, label=label1)
    plt.plot(range(rng), values2, label=label2)
    if val_loss is not None and val_acc is not None:
        plt.plot(range(rng), val_loss, label="val_loss")
        plt.plot(range(rng), val_acc, label="val_acc")
    plt.legend()
    plt.grid()
    plt.show()

def plot_weight(history):
    l1_weight = history["l1"][-1]
    l2_weight = history["l2"][-1]

    sns.heatmap(l1_weight, cmap="bwr")
    plt.show()
    sns.heatmap(l2_weight, cmap="bwr")
    plt.show()

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, roc_auc_score

    data = load_breast_cancer()
    X, y = data.data, data.target

    skh = StratifiedKFold(n_splits=3)
    for train_index, test_index in skh.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        n_input = X_train.shape[1]

        model = AutoMLP(n_input, 0, 2, n_epoch=200, class_weight=y_train)
        model.turning_params(X_train, y_train, X_test, y_test) # L1，L2正則化の強さと中間層の数をチューニング
        history = model.fit(X_train, y_train, val_X=X_test, val_y=y_test, verbose=True)
        results = model.predict(X_test, y_test)

        # 重みの強さをプロット
        plot_weight(history)

        fpr, tpr, thresholds = roc_curve(results["true"], results["pred_prob"])

        # ROC曲線のプロット
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Logistic Regression')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(roc_auc_score(results["true"], results["pred_prob"]))
        plt.show()