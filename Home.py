import streamlit as st
from streamlit_option_menu import option_menu
from cassandra.cluster import Cluster
import pandas as pd
import datetime

# Kết nối đến Cassandra
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('doanbigdata')

st.title("📊 Hệ thống giám sát ô nhiễm không khí")
st.write("Dữ liệu được lấy từ bảng air_quality trong Cassandra")


selected = option_menu(
    menu_title=None,
    options=["Home", "Data Preview"],
    default_index=0,
    orientation="horizontal",
)

if selected == "Home":
    pass

if selected == "Data Preview":
    st.header("Data Preview")
    query = "SELECT * FROM air_quality LIMIT 200"
    rows = session.execute(query)
    df = pd.DataFrame(rows)
    st.dataframe(df)
