import error_generation as eg
import pandas as pd

datasets = ["iris", "cancer", "adult", "abalone", "statlog", "spambase", "heart", "bean"]

# Artificially deteriorates the datasets with a random uniform injection of missing values
for d in datasets:
    data = pd.read_csv("DeterioratedDatasets/"+d+"/"+d+".csv")
    # Inject p% missing values in 5% increments from 5% to 95%
    for p in range(5, 100, 5):
        data_deteriorated = eg.degrade("missing", data, p)  # Create a dataset with p% of missing values
        # Saves the dataset in csv format
        data_deteriorated.to_csv("DeterioratedDatasets/"+d+"/missing/"+d+"_missing_"+str(p)+".csv", index=False)
