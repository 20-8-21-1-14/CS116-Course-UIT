# package import
import streamlit as st
import pandas as pd

# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

# title of the app
st.title("CS116 Web App")
st.markdown("<a style='text-align: center; color: red;'>Made by Hoang Thuan</a>", unsafe_allow_html=True)
# Add a sidebar
st.sidebar.subheader("Settings")

# Setup file upload
uploaded_file = st.sidebar.file_uploader(
                        label="Upload your CSV or Excel file. (200MB max)",
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
    print(non_numeric_columns)
except Exception as e:
    print(e)
    st.write("Please upload file to the application.")
