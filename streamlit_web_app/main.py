# package import
from turtle import home
import streamlit as st
import pandas as pd
import sklearn
import numpy as np 

def main():
    # configuration
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # title of the app
    st.title("CS116 Web App")
    st.markdown("<a style='text-align: center; color: #162bca;'>Made by Hoang Thuan</a>", unsafe_allow_html=True)

    activity = ["Train Test Splitter", 'Home']
    choice = st.sidebar.selectbox("Menu",activity)
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


    # Setup file upload
    uploaded_file = st.file_uploader(
                            label="Upload your dataset in format of CSV or Excel file. (200MB max)",
                            type=['csv', 'xlsx'])

    global df

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            df = pd.read_excel(uploaded_file)

    global numeric_columns
    global non_numeric_columns
    try:
        st.write(df)
        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        non_numeric_columns = list(df.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
    except Exception as e:
        st.write("Please upload file to the application.")

    if choice == 'Home':
        st.subheader("Home page")
        st.markdown("""
			## Very Simple Linear Regression App
		
			##### By **[Ly Hoang Thuan](https://github.com/20-8-21-1-14)**
			""")



if __name__ == '__main__':
	main()