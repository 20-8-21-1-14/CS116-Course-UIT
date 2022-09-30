import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from numpy.random import RandomState
import matplotlib.pyplot as plt

rng = RandomState(19522315)
_rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

def _Simple_LR(dataSrc):
    _read_data = pd.read_csv(dataSrc)
    if _read_data .isnull().sum().any() != 0:
        return 0
    # check data shape
    print(_read_data.shape)
    _train_set = _read_data.sample(frac=0.8, random_state=rng)
    _test_set = _read_data.loc[~_read_data.index.isin(_train_set.index)]
    # print("train:", _train_set.shape)
    # print("test:",_test_set.shape)

    _train_set = _encode_data(_train_set)
    _test_set = _encode_data(_test_set)

    # Separate x and y
    X_train = _train_set.iloc[:,:-1].values
    y_train = _train_set.iloc[:,1].values
    X_test = _test_set.iloc[:,:-1].values
    y_test = _test_set.iloc[:,1].values
    # print(y_train)
    _rf.fit(X_train,y_train)
    y_pred = _rf.predict(X_test)
    _train_score = _rf.score(X_train, y_train)
    _test_score = _rf.score(X_test, y_test)
    # _visualize_result(X_train, y_train, X_test, y_test)
    return _train_score, _test_score

# def _visualize_result(_trainX, _trainY, _testX, _testY):
#     if _trainX.any() == None or _trainY.any()== None or _testX.any() == None:
#         print("Oops! Somethings wrong please check carefully your data!")
#         return
#     fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize= (20, 20))
#     fig.suptitle('Performance train vs test:')
#     plt.scatter(_trainX, _trainY, color="red")
#     plt.title('Linear Regression on training)')
#     plt.plot(_trainX, _rf.predict(_trainX), color="blue", axs=axs[0,0])
#     plt.scatter(_testX, _testY, color="red")
#     plt.title('Linear Regression on testing)')
#     plt.plot(_testX, _rf.predict(_testX), color="blue", axs=axs[0,1])
#     plt.show()
    
#     return

def _encode_data(dataSrc):
    _index_categorical = dataSrc.select_dtypes(include=['object']).columns.tolist()
    # print(_index_categorical)
    if _index_categorical == None:
        return dataSrc

    for _item in _index_categorical:
        if _item == 'Position':
            dataSrc = dataSrc.drop(_item, axis=1)
            continue

        _temp_df = pd.get_dummies(dataSrc[_item])        
        dataSrc = pd.concat([dataSrc, _temp_df], axis=1).reindex(dataSrc.index)
        dataSrc.drop(_item, axis=1, inplace=True)
        # printing df
        print("Process complete your data is now:\n", dataSrc)
    return dataSrc
    
def main():
    _startup_dataScr = "dataset/50_Startups.csv"
    _position_salaries_dataSrc = "dataset\Position_Salaries.csv"
    _salary_data_dataSrc = "dataset\Salary_Data.csv"

    _result_startup_dataScr = _Simple_LR(_startup_dataScr)
    _result_position_salaries_dataScr = _Simple_LR(_position_salaries_dataSrc)
    _result_salary_data_dataScr = _Simple_LR(_salary_data_dataSrc)
    print("Score of 50 Startups dataset: {}.\nScore of Position Salaries dataset: {}.\nScore of Salary Data dataset: {}"
            .format( _result_startup_dataScr, _result_position_salaries_dataScr, _result_salary_data_dataScr))

    return 0

if __name__ == '__main__':
    main()