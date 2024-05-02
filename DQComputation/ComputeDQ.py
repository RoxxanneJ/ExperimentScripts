if __name__ == '__main__':
    freeze_support()
    cpus = cpu_count()
    for error in ['fuzzing']:
        for dataset in ['statlog', 'spambase']:
            path = 'dataset/' + dataset + '/'
            for p in range(5, 55, 5):
                test_trusted = pd.read_csv(path+'trusted_test/'+dataset+'_test.csv')
                y_test_trusted = test_trusted['class'].copy()
                test_trusted.drop(columns=['class'], inplace=True)
                train_trusted = pd.read_csv(path + 'trusted_test/' + dataset + '_train.csv')
                train_trusted = eg.degrade('fuzzing', train_trusted, p)
                train_trusted.to_csv(path + 'trusted_test/' + dataset + '_train_fuzzing_' + str(p) + '.csv',
                                     index=False)
                y_train_trusted = train_trusted['class'].copy()
                train_trusted.drop(columns=['class'], inplace=True)
                dqm.dq_metric_test_para(train_trusted, test_trusted, y_train_trusted, y_test_trusted, crt_names, models,
                                        dataset, 'trusted_test_'+dataset+'_'+error+'_'+str(p))
                del test_trusted
                del y_test_trusted
                del train_trusted
                del y_train_trusted

