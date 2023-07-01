from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import random
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import seaborn as sns
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=FutureWarning)
from model import FBMAD

data = pd.read_csv("finaldataset/sampled_darknet.csv").values
datax = data[:, 1:]
datay = data[:, 0]
random_state = 7
x_train, x_test, y_train, y_test = train_test_split(
    datax, datay, test_size=1500, train_size=6000, random_state=random_state)
x_train, x_valid, y_train, y_valid = train_test_split(
    datax, datay, test_size=1500, train_size=6000, random_state=random_state+2)

x_posl=pd.read_csv("finaldataset/lx_new.csv").values
y_posl=pd.read_csv("finaldataset/ly_new.csv").values
x_posr=pd.read_csv("finaldataset/rx_new.csv").values
y_posr=pd.read_csv("finaldataset/ry_new.csv").values
x_poss=pd.read_csv("finaldataset/svmx_new.csv").values
y_poss=pd.read_csv("finaldataset/svmy_new.csv").values

x_pos = np.row_stack((x_posl, x_posr, x_poss))
y_pos = np.row_stack((y_posl, y_posr, y_poss)).ravel()

x_trainsvm, _, y_trainsvm, _ = train_test_split(
    datax, datay, test_size=500, train_size=2000, random_state=random_state)
x_svm = np.row_stack((x_trainsvm, x_poss))
y_svm = np.append(y_trainsvm, y_poss)

x_trainridge, _, y_trainridge, _ = train_test_split(
    datax, datay, test_size=500, train_size=2000, random_state=random_state+1)
x_ridge = np.row_stack((x_trainridge, x_posr))
y_ridge = np.append(y_trainridge, y_posr)

x_trainlog, _, y_trainlog, _ = train_test_split(
    datax, datay, test_size=500, train_size=2000, random_state=random_state+2)
x_log = np.row_stack((x_trainlog, x_posl))
y_log = np.append(y_trainlog, y_posl)

x_trainmuti, _, y_trainmuti, _ = train_test_split(
    datax, datay, test_size=500, train_size=2000, random_state=random_state)
x_posm, _, y_posm, _ = train_test_split(
    x_pos, y_pos, test_size=1000, train_size=500, random_state=random_state)
x_muti = np.row_stack((x_trainsvm, x_posm))
y_muti = np.append(y_trainsvm, y_posm)

x_trainmuti, _, y_trainmuti, _ = train_test_split(
    datax, datay, test_size=500, train_size=2000, random_state=random_state+1)
x_posm, _, y_posm, _ = train_test_split(
    x_pos, y_pos, test_size=1000, train_size=500, random_state=random_state+1)
x_muti2 = np.row_stack((x_trainsvm, x_posm))
y_muti2 = np.append(y_trainsvm, y_posm)

x_trainmuti, _, y_trainmuti, _ = train_test_split(
    datax, datay, test_size=500, train_size=2000, random_state=random_state+2)
x_posm, _, y_posm, _ = train_test_split(
    x_pos, y_pos, test_size=1000, train_size=500, random_state=random_state+2)
x_muti3 = np.row_stack((x_trainsvm, x_posm))
y_muti3 = np.append(y_trainsvm, y_posm)

x_trainmuti, _, y_trainmuti, _ = train_test_split(
    datax, datay, test_size=500, train_size=2000, random_state=random_state+3)
x_posm, _, y_posm, _ = train_test_split(
    x_pos, y_pos, test_size=1000, train_size=500, random_state=random_state+3)
x_muti4 = np.row_stack((x_trainsvm, x_posm))
y_muti4 = np.append(y_trainsvm, y_posm)


fbmad = FBMAD(x_ridge, y_ridge.ravel(), x_valid, y_valid.ravel())
fbmad.train_by_epoch(test_x=x_test, test_y=y_test.ravel(), size=50)
