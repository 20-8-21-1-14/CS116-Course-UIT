# package import
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, log_loss
from sklearn.decomposition import PCA

def _KFold(X_data, Y_data, measurement, k_val, model_type):
    my_kfold = KFold(int(k_val))
    my_kfold.get_n_splits(X_data)
    lr = LinearRegression()
    lg = LogisticRegression()
    train_error = []
    test_error = []
    st.markdown("<a style='text-align: center; color: #27b005;'>Running...</a>", unsafe_allow_html=True)
    for train, test in my_kfold.split(X_data):
        X_train, X_test = X_data[train], X_data[test]
        Y_train, Y_test = Y_data[train], Y_data[test]
        if model_type == 'Linear':
            lr.fit(X_train, Y_train)
            y_train_data_pred = lr.predict(X_train)
            y_test_data_pred = lr.predict(X_test) 
        if model_type == 'Logistic':
            lg.fit(X_train, Y_train)
            y_train_data_pred = lg.predict(X_train)
            y_test_data_pred = lg.predict(X_test)

        if measurement =='MSE':
            train_error.append(mean_squared_error(Y_train, y_train_data_pred))
            test_error.append(mean_squared_error(Y_test, y_test_data_pred))
        elif  measurement =='MAE':
            train_error.append(mean_absolute_error(Y_train, y_train_data_pred))
            test_error.append(mean_absolute_error(Y_test, y_train_data_pred))
        elif measurement == 'Log Loss':
            train_error.append(log_loss(Y_train, y_train_data_pred))
            test_error.append(log_loss(Y_test, y_train_data_pred))
        elif measurement == 'F1 Score':
            train_error.append(f1_score(Y_train, y_train_data_pred))
            test_error.append(f1_score(Y_test, y_train_data_pred))

    #draw hist for k-fold
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,2,1)
    ax.bar(range(1, my_kfold.get_n_splits() + 1), np.array(train_error).ravel(), color ='r')
    ax.set_xlabel('number of fold')
    ax.set_ylabel('Training error')
    ax.set_title('Training error across folds')

    ax2 = fig2.add_subplot(1,2,2)
    ax2.bar(range(1, my_kfold.get_n_splits() + 1), np.array(test_error).ravel())
    ax2.set_xlabel('number of fold')
    ax2.set_ylabel('Testing error')
    ax2.set_title('Testing error across folds')

    st.pyplot(fig2)
    if measurement == 'Log Loss':
        st.write("Log loss with KFold on Training is: ", test_error)
        st.write("Test set Log loss with KFold is: ", test_error)
    if measurement == 'F1 Score':
        st.write("F1 Score with KFold on train set is: ", train_error)
        st.write("F1 Score with KFold on test set is: ", test_error)

def train_test_Split(X, Y, split_ratio):
    if X is None or split_ratio is None:
        st.write('Something wrong!')
        return -1
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=split_ratio[0])
    return X_train, X_test, Y_train, Y_test

def _predictor(X_train, X_test, Y_train, Y_test, predictor="Linear", measurement='MSE'):
    # default func linear regression
    if predictor=='Linear':
        train_error = 0
        test_error = 0
        regressor = LinearRegression()
        regressor.fit(X_train, Y_train)

        Y_pred_train = regressor.predict(X_train)
        Y_pred_test = regressor.predict(X_test)
        if measurement =='MSE':
            test_error = mean_squared_error(Y_test, Y_pred_test)
            train_error =  mean_squared_error(Y_train, Y_pred_train)

        if measurement =='MAE':
            print('Y_test', Y_test)
            print('Y_pred_test', Y_pred_test)
            test_error = mean_absolute_error(Y_test, Y_pred_test)
            train_error =  mean_absolute_error(Y_train, Y_pred_train)
        
        n=1
        r = np.arange(n)
        width = 0.5
        fig = plt.figure()
        plt.bar(r, train_error, color = 'r', width = width, edgecolor = 'black', label='Train Error')
        plt.bar(r + width, test_error, color = 'g', width = width, edgecolor = 'black', label='Test Error')
        plt.title("Train and Test Error")
        plt.legend()

        st.pyplot(fig)
    
        return train_error, test_error
    if predictor=='Logistic':
        train_error = 0
        test_error = 0
        regressor = LogisticRegression()
        regressor.fit(X_train, Y_train)

        Y_pred_train = regressor.predict_proba(X_train)
        Y_pred_test = regressor.predict_proba(X_test)
        if measurement == 'Log Loss':
            test_error = log_loss(Y_test, Y_pred_test)
            train_error =  log_loss(Y_train, Y_pred_train)
            st.write("Log loss on Training is: ", test_error)
            st.write("Test set Log loss is: ", test_error)

        if measurement =='F1 Score':
            test_error = f1_score(Y_test, Y_pred_test)
            train_error =  f1_score(Y_train, Y_pred_train)
            st.write("Log loss on Training is: ", test_error)
            st.write("Test set Log loss is: ", test_error)
        return train_error, test_error
    else:
        st.write("Something went wrong!!!")

def PCA_Process(_dataset, n_component, test_set=False):
    st.subheader("PCA - Principal component analysis")
    if n_component != None and n_component > 0:
        try:
            pca = PCA(n_component)
            train_component = pca.fit_transform(_dataset)
            if test_set:
                tst_component = pca.transform(_dataset)
                return tst_component
        except Exception as e:
            raise Exception("Somethings went wrong please try again!")

    return train_component

def main():
    # configuration
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # title of the app
    st.title("CS116 Web App")
    st.markdown("<a style='text-align: center; color: #162bca;'>Made by Hoang Thuan</a>", unsafe_allow_html=True)

    activity = ['Home', 'Visualize data', 'Predictor', 'PCA']
    choice = st.sidebar.selectbox("Menu",activity)         

    if choice == 'Home':
        st.subheader("Home page")
        st.markdown("""
			## Very Simple Linear Regression App
		
			##### By **[Ly Hoang Thuan](https://github.com/20-8-21-1-14)**
			""")

    global df

    if choice == 'Visualize data':
        # Setup file upload
        uploaded_file = st.file_uploader(
                                label="Upload your dataset in format of CSV or Excel file. (200MB max)",
                                type=['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                try:
                    df = pd.read_csv(uploaded_file,sep=';')
                except:
                    df = pd.read_csv(uploaded_file)

                object_data_lst = list(df.select_dtypes(include=['object']).columns)
                for col_name in object_data_lst:
                    df[col_name] = df[col_name].astype('category')
                    df[col_name] = df[col_name].cat.codes

                X_data = df.iloc[:, 0:-1];
                Y_data = df.iloc[:, -1:];
            except Exception as e:
                df = pd.read_excel(uploaded_file)
                X_data = df.iloc[:, 0:-1];
                Y_data = df.iloc[:, -1:];
        global numeric_columns
        global non_numeric_columns
        try:
            st.write(df)
            numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
            non_numeric_columns.append(None)
        except Exception as e:
            st.write("Please upload file to the application.")

    if choice == 'Predictor':
        # Setup file upload
        uploaded_file = st.file_uploader(
                                label="Upload your dataset in format of CSV or Excel file. (200MB max)",
                                type=['csv', 'xlsx'])
        _data_name = st.radio('Are you using wine dataset?', ('Yes', 'No'))
        if uploaded_file is not None:
            try:
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                    df = pd.read_csv(uploaded_file, sep=';')

                if _data_name == 'Yes':
                    # Y_data = df['1']
                    # X_data = df.drop('1', axis=1)
                    X_data = df.iloc[:, 1:].copy(deep=True);
                    Y_data = df.iloc[:, :1].copy(deep=True);

                else:
                    object_data_lst = list(df.select_dtypes(include=['object']).columns)
                    for col_name in object_data_lst:
                        df[col_name] = df[col_name].astype('category')
                        df[col_name] = df[col_name].cat.codes

                    X_data = df.iloc[:, 0:-1].copy(deep=True);
                    Y_data = df.iloc[:, -1:].copy(deep=True);
            except Exception as e:
                if _data_name == 'Yes':
                    X_data = df.iloc[:, 1:].copy(deep=True);
                    Y_data = df.iloc[:, :1].copy(deep=True);
                else:
                    df = pd.read_excel(uploaded_file)
                    X_data = df.iloc[:, 0:-1].copy(deep=True);
                    Y_data = df.iloc[:, -1:].copy(deep=True);

        _preprocess_option = st.selectbox('How would you like to be process your data?', ('Feature Picker', 'PCA'))
        if  _preprocess_option == "Feature Picker": 
            st.subheader("Feature Picker")
            temp = []
            try:
                col_name = []
                for col in X_data.columns:
                    col_name.append(col)
                fea_option = st.multiselect('Select feature to keep:',
                            [item for item in col_name])
                
                for col in X_data.columns:
                    if col not in fea_option:
                        temp.append(col)
                
                X_data.drop(temp, axis=1, inplace=True)
                if X_data is not None:
                    st.write(X_data.head(10))
            except Exception as e:
                st.write("Something wrong!")

        predictor = st.selectbox('Which model do you like?', ['Linear', 'Logistic', 'Other'])
        st.write('You choose', predictor)

        st.subheader("Train Test Determine")

        _deter = st.radio(
                "How you want your data to use?",
                ('Train test split', 'K-fold'))
        if _deter == 'Train test split':
                try:
                    train_size = st.number_input('Insert train set size', min_value=0.0, max_value=0.9, step=0.1)
                    test_size = 1.0 - train_size
                    st.write('The current train size and test size is: ', train_size, test_size)
                except Exception as e:
                    if train_size < test_size:
                        raise Exception("Sorry, Test size is bigger than train size!!")
                    else:
                        raise Exception("Somethings went wrong please try again!")

        if _deter == 'K-fold':
            try:
                k_num = st.number_input("K value for K-fold validation:", min_value=2, max_value=10, step=1)
            except Exception as e:
                    st.error('Please pick k value!', icon="🚨")
        if predictor=='Linear':
            measurement = st.selectbox('Which measurement do you like?', ['MSE', 'MAE'])
        else:
            measurement =  st.selectbox('Which measurement do you like?', ['Log Loss', 'F1 Score'])
        st.write('You choose', measurement)

        if st.button('RUN'):
            try:
                if _deter == 'K-fold':
                    if predictor=='Linear' or predictor=='Logistic':
                        try:
                            X_data = X_data.to_numpy()
                            Y_data= Y_data.to_numpy()
                            _KFold(X_data, Y_data, measurement=measurement, k_val=k_num, model_type=predictor)
                        except Exception as e:
                            print(e)
                            st.error('Something wrong!', icon="🚨")

                if _deter == 'Train test split':
                    if predictor == 'Linear' or predictor=='Logistic':
                        try:
                            X_data = X_data.to_numpy()
                            Y_data= Y_data.to_numpy()
                            X_train, X_test, Y_train, Y_data = train_test_Split(X_data, Y_data, split_ratio=[train_size, test_size])
                            _result_ln = _predictor(X_train, X_test, Y_train, Y_data, predictor, measurement)
                            st.write('Your Linear regression performance (Train , Test):', _result_ln)
                        except Exception as e:
                            print(e)
                            st.error('Something wrong!', icon="🚨")
                    else:
                        st.error('We are working on this!')
            except Exception as e:
                st.markdown("<a style='text-align: center; color: #bf4c04;'>You are missing some thing please recheck again!</a>", unsafe_allow_html=True)
        else:
            st.markdown("<a style='text-align: center; color: #27b005;'>Hit run to predict.</a>", unsafe_allow_html=True)

    if choice == 'PCA':
        # Setup file upload
        uploaded_file = st.file_uploader(
                                label="Upload your dataset in format of CSV or Excel file. (200MB max)",
                                type=['csv', 'xlsx'])
        if uploaded_file is not None:
            _data_name = st.radio('Are you using wine dataset?', ('Yes', 'No'))

        if uploaded_file is not None:
            try:
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                    df = pd.read_csv(uploaded_file, sep=';')

                if _data_name == 'Yes':
                    # Y_data = df['1']
                    # X_data = df.drop('1', axis=1)
                    X_data = df.iloc[:, 1:].copy(deep=True);
                    Y_data = df.iloc[:, :1].copy(deep=True);

                else:
                    object_data_lst = list(df.select_dtypes(include=['object']).columns)
                    for col_name in object_data_lst:
                        df[col_name] = df[col_name].astype('category')
                        df[col_name] = df[col_name].cat.codes

                    X_data = df.iloc[:, 0:-1].copy(deep=True);
                    Y_data = df.iloc[:, -1:].copy(deep=True);
            except Exception as e:
                if _data_name == 'Yes':
                    X_data = df.iloc[:, 1:].copy(deep=True);
                    Y_data = df.iloc[:, :1].copy(deep=True);
                else:
                    df = pd.read_excel(uploaded_file)
                    X_data = df.iloc[:, 0:-1].copy(deep=True);
                    Y_data = df.iloc[:, -1:].copy(deep=True);

        predictor = st.selectbox('Which model do you like?', ['Linear', 'Logistic', 'Other'])
        st.write('You choose', predictor)

        st.subheader("Train Test Determine")

        _deter = st.radio(
                "How you want your data to use?",
                ('Train test split', 'K-fold'))
        if _deter is not None:
            st.subheader("Input number of components to keep!")
            n_components = st.number_input('Insert a number of components:', min_value =1, max_value=4, step=1)

        if _deter == 'Train test split':
                try:
                    train_size = st.number_input('Insert train set size', min_value=0.0, max_value=0.9, step=0.1)
                    test_size = 1.0 - train_size
                    st.write('The current train size and test size is: ', train_size, test_size)
                except Exception as e:
                    if train_size < test_size:
                        raise Exception("Sorry, Test size is bigger than train size!!")
                    else:
                        raise Exception("Somethings went wrong please try again!")

        if _deter == 'K-fold':
            try:
                k_num = st.number_input("K value for K-fold validation:", min_value=2, max_value=10, step=1)
            except Exception as e:
                    st.error('Please pick k value!', icon="🚨")
        if predictor=='Linear':
            measurement = st.selectbox('Which measurement do you like?', ['MSE', 'MAE'])
        else:
            measurement =  st.selectbox('Which measurement do you like?', ['Log Loss', 'F1 Score'])
        st.write('You choose', measurement)

        if st.button('RUN'):
            try:
                if _deter == 'K-fold':
                    if predictor=='Linear' or predictor=='Logistic':
                        try:
                            X_data = X_data.to_numpy()
                            Y_data= Y_data.to_numpy()
                            X_data = PCA_Process(X_train, n_components, False)
                            # X_test = PCA_Process(X_test, n_components, test_set=True)
                            _KFold(X_data, Y_data, measurement=measurement, k_val=k_num, model_type=predictor)
                        except Exception as e:
                            print(e)
                            st.error('Something wrong!', icon="🚨")

                if _deter == 'Train test split':
                    if predictor == 'Linear' or predictor=='Logistic':
                        try:
                            X_train, X_test, Y_train, Y_data = train_test_Split(X_data, Y_data, split_ratio=[train_size, test_size])
                            X_train = PCA_Process(X_train, n_components, False)
                            X_test = PCA_Process(X_test, n_components, test_set=True)
                            X_data = X_data.to_numpy()
                            Y_data= Y_data.to_numpy()
                            _result_ln = _predictor(X_train, X_test, Y_train, Y_data, predictor, measurement)
                            st.write('Your Linear regression performance (Train , Test):', _result_ln)
                        except Exception as e:
                            print(e)
                            st.error('Something wrong!', icon="🚨")
                    else:
                        st.error('We are working on this!')
            except Exception as e:
                st.markdown("<a style='text-align: center; color: #bf4c04;'>You are missing some thing please recheck again!</a>", unsafe_allow_html=True)
        else:
            st.markdown("<a style='text-align: center; color: #27b005;'>Hit run to predict.</a>", unsafe_allow_html=True)


if __name__ == '__main__':
	main()