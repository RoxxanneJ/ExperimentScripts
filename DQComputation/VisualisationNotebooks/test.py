import numpy as np
import os
import pandas as pd

def load_cimb_score(dataset):
    path = "../Output/class balance/scores/"
    end = "_(x,qa,qf,time).npy"
    qas = []
    qfs = []
    times = []
    if dataset == 'iris':
        _, qa, qf, time = np.load(path + dataset + "_3class_1_1_(x,y,y1,y2,z,z1,z2,time).npy", allow_pickle=True)
    elif dataset in ['adult', 'cancer']:
        _, qa, qf, time = np.load(path + dataset + "_1_1_(x,y,y1,y2,z,z1,z2,time).npy", allow_pickle=True)
    else:
        _, qa, qf, time = np.load(path + dataset + "_0" + end, allow_pickle=True)
    qas.append(qa)
    qfs.append(qf)
    times.append(time)
    for p in range(5, 100, 5):
        if os.path.isfile(path + dataset + "_" + str(p) + end):
            _, qa_p, qf_p, time_p = np.load(path + dataset + "_" + str(p) + end, allow_pickle=True)
            qas.append(qa_p)
            qfs.append(qf_p)
            times.append(time_p)
        else:
            qas.append((np.nan)*3)
            qfs.append((np.nan)*3)
            times.append(np.nan)
    return np.array(qas), np.array(qfs), np.array(times)

iris_qas, iris_qfs, iris_times = load_cimb_score('iris')
cancer_qas, cancer_qfs, cancer_times = load_cimb_score('cancer')
adult_qas, adult_qfs, adult_times = load_cimb_score('adult')
heart_qas, heart_qfs, heart_times = load_cimb_score('heart')
abalone_qas, abalone_qfs, abalone_times = load_cimb_score('abalone')
statlog_qas, statlog_qfs, statlog_times = load_cimb_score('statlog')
spambase_qas, spambase_qfs, spambase_times = load_cimb_score('spambase')
bean_qas, bean_qfs, bean_times = load_cimb_score('bean')

np.save("save/iris_imbalanced_(qa,qf,time).npy", (iris_qas, iris_qfs, iris_times))
np.save("save/cancer_imbalanced_(qa,qf,time).npy", (cancer_qas, cancer_qfs, cancer_times))
np.save("save/adult_imbalanced_(qa,qf,time).npy", (adult_qas, adult_qfs, adult_times))
np.save("save/heart_imbalanced_(qa,qf,time).npy", (heart_qas, heart_qfs, heart_times))
np.save("save/abalone_imbalanced_(qa,qf,time).npy", (abalone_qas, abalone_qfs, abalone_times))
np.save("save/statlog_imbalanced_(qa,qf,time).npy", (statlog_qas, statlog_qfs, statlog_times))
np.save("save/spambase_imbalanced_(qa,qf,time).npy", (spambase_qas, spambase_qfs, spambase_times))
np.save("save/bean_imbalanced_(qa,qf,time).npy", (bean_qas, bean_qfs, bean_times))


