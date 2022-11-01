# package import
import streamlit as st
import pandas as pd

# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

# title of the app
st.title("CS116 Web App")
st.markdown("<a style='text-align: center; color: #162bca;'>Made by Hoang Thuan</a>", unsafe_allow_html=True)

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
