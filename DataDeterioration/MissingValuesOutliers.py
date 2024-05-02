import error_generation as eg
import pandas as pd

datasets = ["iris", "cancer", "adult", "abalone", "statlog", "spambase", "heart", "bean"]
errors = ["missing", "outlier"]

# Artificially deteriorates the datasets with a random uniform injection of missing values
for d in datasets:
    data = pd.read_csv("DeterioratedDatasets/"+d+"/"+d+".csv")
    # Inject p/2% of missing values and P/2% of outliers in 5% increments from 5% to 95% in disjointed parts of the data
    for p in range(5, 100, 5):
        data_deteriorated = eg.gen_dataset(data, errors, p)
        # Saves the dataset in csv format
        data_deteriorated.to_csv("DeterioratedDatasets/"+d+"/missing+outlier/"+d+"_missing_outlier_"+str(p)+".csv",
                                 index=False)
