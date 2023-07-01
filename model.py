import numpy as np
from sklearn.metrics import f1_score
from sklearn import linear_model, svm, neighbors
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
class FBMAD:
    def __init__(
        self, train_x, train_y, valid_x, valid_y, mean=100, default_model=True
    ):
        self.models = []
        self.model_count = 0
        self.mean = mean
        self.conf = []  
        self.count_N = []
        self.count_NI = []
        self.count_renew = 0

        if default_model:
            self.add_model(linear_model.LogisticRegression())
            self.add_model(
                svm.SVC(
                    gamma="scale", C=1.0, decision_function_shape="ovr", kernel="rbf"
                )
            )
            self.add_model(DecisionTreeClassifier())
            self.add_model(neighbors.KNeighborsClassifier(n_neighbors=5))

        self.init_all_model(train_x, train_y, valid_x, valid_y)

    def get_random(self):
        NI = 0
        mean = self.mean
        conv = int(self.mean / 10)
        while NI <= 1:
            NI = np.random.normal(2 * mean, conv, 1)
        return int(NI)

    def add_model(self, new_model):
        self.models.append(new_model)
        self.model_count += 1
        self.conf.append(0)  
        self.count_N.append(0)
        self.count_NI.append(self.get_random())

    def init_all_model(self, train_x, train_y, valid_x, valid_y):
        for i, model in enumerate(self.models):
            model.fit(train_x, train_y)
            self.conf[i] = f1_score(valid_y, model.predict(valid_x))  

    def __call__(self, test_x):
        return self.predict(test_x)

    def predict(self, test_x):
        predict_list = []
        N = np.array(self.count_N)
        NI = np.array(self.count_NI)
        mu = NI / (N + NI)
        gamma = self.conf  
        for model in self.models:
            predict_list.append(model.predict(test_x))
        predict_mat = np.array(predict_list).reshape(self.model_count, -1)
        weight = np.array(gamma).reshape(1, -1)
        weight_sum = np.sum(weight)
        result = (weight @ predict_mat) / weight_sum
        result = result >= 0.5
        return result.reshape(
            -1,
        )

    def score_of_normal_dhr(self, test_x, test_y):
        predict_list = []
        for model in self.models:
            predict_list.append(model.predict(test_x))
        predict_mat = np.array(predict_list).reshape(self.model_count, -1)
        weight = np.ones_like(np.array(self.conf))  
        weight_sum = np.sum(weight)
        result = (weight @ predict_mat) / weight_sum
        result = result > 0.5
        test_y = test_y.ravel()
        score = sum(result == test_y) / test_y.shape[0]
        return score

    def score_each_model(self, test_x, test_y):
        new_conf = []  
        for model in self.models:
            new_conf.append(f1_score(test_y, model.predict(test_x)))
        return new_conf

    def renew_conf(self, test_x, test_y, alpha=0.5):
        i = 0
        N = np.array(self.count_N)
        NI = np.array(self.count_NI)
        mu = NI / (N + NI)
        for model in self.models:
            new_score = f1_score(test_y, model.predict(test_x))
            self.conf[i] = new_score  
            if self.conf[i] * mu[i] < alpha:  
                self.count_renew += 1
                self.count_N[i] = 0
                self.count_NI[i] = self.get_random()
                N[i] = 0
                NI[i] = self.get_random()
                self.conf[i] = f1_score(test_y, model.predict(test_x))  
            i += 1

    def train_by_epoch(self, test_x, test_y, size=20, alpha=0.5):
        length, _ = test_x.shape
        num = int(length / size)
        conf_list = []
        conf_dhr = []
        conf_norm = []
        gamma_model = []
        for i in range(num):
            batch_test_x = test_x[i * size : (i + 1) * size, :]
            batch_test_y = test_y[i * size : (i + 1) * size]
            new_conf = self.score_each_model(batch_test_x, batch_test_y)
            conf_list.append(new_conf)
            conf_dhr.append(f1_score(batch_test_y, self.predict(batch_test_x)))
            conf_norm.append(self.score_of_normal_dhr(batch_test_x, batch_test_y))
            for j in range(len(self.count_N)):
                self.count_N[j] += size
            N = np.array(self.count_N)
            NI = np.array(self.count_NI)
            mu = NI / (N + NI)
            gamma_model.append(new_conf * mu)
            self.renew_conf(batch_test_x, batch_test_y, alpha)
        # plot
        conf_list = np.array(conf_list).T
        gamma_model = np.array(gamma_model).T
        axis_x = np.arange(0, num)
        threshold = np.ones(num) * alpha
        plotn = 15
        best = np.max(conf_list, axis=0)
        worst = np.min(conf_list, axis=0)
        average = np.average(conf_list, axis=0)
        
        d = {'epoch': axis_x[:plotn], 'best': best[:plotn], 'worst': worst[:plotn], 'average': average[:plotn], 'norm dhr': conf_norm[:plotn], 'bayes dhr': conf_dhr[:plotn]}
        data_preproc = pd.DataFrame(d)
        sns.lineplot(x='epoch', y='value', hue='variable', data=pd.melt(data_preproc, ['epoch']))
        print(data_preproc.values)
        plt.savefig("pic/performance.png", dpi=300)
        d = {'epoch': axis_x, 'logistic': gamma_model[0, :], 'svm': gamma_model[1, :], 'decision tree': gamma_model[2,
                                                                                                                    :], 'knn': gamma_model[3, :], 'threshold':threshold}
        data_preproc = pd.DataFrame(d)
        plt.clf()
        sns.lineplot(x='epoch', y='value', hue='variable',
                     data=pd.melt(data_preproc, ['epoch']))
        plt.savefig("pic/submodel.png", dpi=300)

        print(data_preproc.values)
        print("renew frequency:" + str(self.count_renew / num))
        return self.count_renew
