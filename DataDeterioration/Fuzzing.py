import error_generation as eg
import pandas as pd

datasets = ["iris", "cancer", "adult", "abalone", "statlog", "spambase", "heart", "bean"]

# Artificially deteriorates the datasets with a random uniform injection of fuzzing
for d in datasets:
    data = pd.read_csv("DeterioratedDatasets/"+d+"/"+d+".csv")
    # Inject p% fuzzing in 5% increments from 5% to 95%
    for p in range(5, 100, 5):
        nb_to_modify = round(data.shape[0] * (p / 100))  # Percentage of lines not cells this time
        data_deteriorated = eg.degrade("fuzzing", data, p, nb_to_modify)  # Create a dataset with p% of fuzzing
        # Saves the dataset in csv format
        data_deteriorated.to_csv("DeterioratedDatasets/"+d+"/fuzzing/"+d+"_fuzzing_"+str(p)+".csv", index=False)