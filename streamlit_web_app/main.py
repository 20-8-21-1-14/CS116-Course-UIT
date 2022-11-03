# package import
from turtle import home
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_With_KFold(X_data_np, Y_data_np, model):
    st.subheader("Salary Determination")
    k_val = st.slider("K value for K-fold validation:", min_value=2, max_value=10, step=1)

    kfold = KFold(k_val, True, 1);
    accuracy_list = [];
    for train, test in kfold.split(X_data_np, Y_data_np):
        X_train, X_test = X_data_np[train], X_data_np[test];
        Y_train, Y_test = Y_data_np[train], Y_data_np[test]

        model.fit(X_train, Y_train);
        Y_pred = model.predict(X_test);
        accuracy_list.append(accuracy_score(Y_test, Y_pred));

    return np.mean(accuracy_list);

def train_test_Split(dataset, split_ratio):
    if dataset is None or split_ratio is None:
        st.write('Something wrong!')
        return -1
    df = pd.DataFrame(dataset)
    X_data = df.iloc[:, 0:-1];
    Y_data = df.iloc[:, -1:];
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size=split_ratio[0])
    return X_train, X_test, Y_train, Y_test

def _predictor(X_train, X_test, Y_train, Y_test, predictor="linear", measurement='MSE'):
    # default func linear regression
    if predictor=='linear':
        regressor = LinearRegression()
        regressor.fit(X_train, Y_train)

        Y_pred_train = regressor.predict(X_train)
        Y_pred = regressor.predict(X_test)
        if measurement =='MSE':
            test_MSE = mean_squared_error(Y_test, Y_pred)
            train_MSE =  mean_squared_error(Y_train, Y_pred_train)
            return [train_MSE, test_MSE]
        if measurement =='MAE':
            test_MAE = mean_absolute_error(Y_test, Y_pred)
            train_MAE =  mean_absolute_error(Y_train, Y_pred_train)
            return [train_MAE, test_MAE]
    else:
        st.write("Something went wrong!!!")

def main():
    # configuration
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # title of the app
    st.title("CS116 Web App")
    st.markdown("<a style='text-align: center; color: #162bca;'>Made by Hoang Thuan</a>", unsafe_allow_html=True)

    activity = ['Home', 'Visualize data', 'Feature Pick','Train Test Splitter','K-fold', 'Predictor']
    choice = st.sidebar.selectbox("Menu",activity)
    # Setup file upload
    uploaded_file = st.file_uploader(
                            label="Upload your dataset in format of CSV or Excel file. (200MB max)",
                            type=['csv', 'xlsx'])
    
    global df

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            X_data = df.iloc[:, 0:-1];
            Y_data = df.iloc[:, -1:];
        except Exception as e:
            print(e)
            df = pd.read_excel(uploaded_file)
            X_data = df.iloc[:, 0:-1];
            Y_data = df.iloc[:, -1:];
            

    if choice == 'Home':
        st.subheader("Home page")
        st.markdown("""
			## Very Simple Linear Regression App
		
			##### By **[Ly Hoang Thuan](https://github.com/20-8-21-1-14)**
			""")

    if choice == 'Visualize data':
        global numeric_columns
        global non_numeric_columns
        try:
            st.write(df)
            numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
            non_numeric_columns.append(None)
        except Exception as e:
            st.write("Please upload file to the application.")

    if choice == 'Feature Pick':
        st.subheader("Feature Picker")
        try:
            col_name = []
            for col in X_data.columns:
                col_name.append(col)
            fea_option = st.multiselect('Select feature to keep:',
                        [item for item in col_name])
                        
            if len(fea_option) != 0:
                st.write('Here is your feature:', fea_option)
            
        except Exception as e:
            st.write("Something wrong !")

        temp = []
        for col in X_data.columns:
            if col not in fea_option:
                temp.append(col)

        # print(temp)
        X_data.drop(temp, axis=1, inplace=True)
        # print(X_data.head(5))

    if choice == 'K-fold':
        st.markdown("<h3 style='text-align: center; color: #2596be;'>K-fold Validator</a>", unsafe_allow_html=True)
        try:
            print("Test")
        except Exception as e:
            print("Test")

    if choice == "Train Test Splitter":
        st.subheader("Train Test Splitter")
        try:
            train_size = st.number_input('Insert test set size', min_value=0.0, max_value=0.9, step=0.1)
            test_size = 1.0-train_size
            st.write('The current train size and test size is: ', train_size, test_size)
        except Exception as e:
            if train_size < test_size:
                raise Exception("Sorry, Test size is bigger than train size!!")
            else:
                raise Exception("Somethings went wrong please try again!")


    if choice == 'Predictor':
        predictor = st.selectbox('Which model do you like?', ['Linear', 'Other'])
        measurement = st.selectbox('Which measurement do you like?', ['MSE', 'MAE'])
        st.write('You choose', measurement)

        X_train, X_test, Y_train, Y_data = train_test_Split(X_data, Y_data, split_ratio=[train_size,test_size])
        _result = _predictor(X_train, X_test, Y_train, Y_data, predictor, measurement)
        


if __name__ == '__main__':
	main()