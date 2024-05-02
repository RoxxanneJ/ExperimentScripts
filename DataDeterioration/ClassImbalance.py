# depuis le train balanced on suprime 5% de la classe minoritaire (vérifier laquelle)
import pandas as pd
import numpy as np

datasets = ["iris", "cancer", "adult", "heart", "spambase", "statlog", "abalone", "bean"]
# The minority class we use to create imbalance in each dataset
minority_class = [1, 1, 1, 0, 1, 0, 1, 5]

for d, m in zip(datasets, minority_class):
    for p in range(5, 100, 5):
        data = pd.read_csv("../DataDeterioration/DeterioratedDatasets/" + d + "/imbalanced/" + d +
                           "_train_balanced.csv")
        minority_class = data[data['class'] == m].copy()  # We isolate the minority class
        data.drop(index=minority_class.index, inplace=True)
        tot_class_rows = minority_class.shape[0]
        nb_to_remove = int(np.ceil(p * tot_class_rows / 100))  # We remove p% of the total number of rows of the class
        removed_rows = minority_class.sample(n=nb_to_remove, replace=False, axis='index')
        minority_class.drop(index=removed_rows.index, inplace=True)  # We remove rows from the minority class
        imbalanced_df = pd.concat([data, minority_class])  # We concatenate with the rest of the data again
        imbalanced_df.to_csv("../DataDeterioration/DeterioratedDatasets/"+d+"/imbalanced/"+d+"_train_"+str(p)+".csv",
                             index=False)
