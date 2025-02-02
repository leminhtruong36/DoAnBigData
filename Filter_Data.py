import streamlit as st
from cassandra.cluster import Cluster
from streamlit_option_menu import option_menu
import pandas as pd
import datetime

# Kết nối đến Cassandra
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('doanbigdata')

# Giao diện Streamlit
st.title("📊 Hệ thống giám sát ô nhiễm không khí")
st.write("Dữ liệu được lấy từ bảng air_quality trong Cassandra")

with st.sidebar:
    selected = option_menu(
        menu_title="Filter Data",
        options=["Date", "CO(GT)", "NOx(GT)", "NO2(GT)", "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"],
        default_index=0,
        orientation="vertical",
    )
if selected == "Date":
    st.subheader("Lọc dữ liệu theo ngày")
    start_date = st.date_input("Chọn ngày bắt đầu", datetime.date(2004, 3, 10), min_value=datetime.date(2004, 3, 10), max_value=datetime.date(2005, 4, 4))
    end_date = st.date_input("Chọn ngày kết thúc", datetime.date(2004, 3, 10), min_value=datetime.date(2004, 3, 10), max_value=datetime.date(2005, 4, 4))
    if st.button("Lọc dữ liệu"):
        # Kiểm tra điều kiện lọc
        if start_date > end_date:
            st.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc")
        if start_date <= end_date:
            #st.subheader(f"Du lieu tu ngay {start_date} den ngay {end_date}")
            query = f"SELECT * FROM air_quality WHERE date >= '{start_date}' AND date <= '{end_date}' ALLOW FILTERING"
            # Thực thi truy vấn
            rows = session.execute(query)
            df = pd.DataFrame(rows)
            # Hiển thị dữ liệu
            st.dataframe(df)
        else:
            st.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc")
    if st.button("Reset", type="primary"):
        pass

if selected == "CO(GT)":
    selected = option_menu(
        menu_title=None,
        options=["Khoảng giá trị", "Giá trị cố định"],
        default_index=0,
        orientation="horizontal",
    )
    if selected == "Khoảng giá trị":
        st.subheader("Lọc dữ liệu theo khoảng giá trị CO(GT)")
        co = st.slider("Chọn khoảng giá trị CO(GT)", min_value=0.0, max_value=12.0, value=(0.0, 12.0), step=0.1)
        if st.button("Lọc dữ liệu"):
            #st.subheader(f"Du lieu co gia tri CO(GT) tu {co[0]} den {co[1]}")
            query = f"SELECT * FROM air_quality WHERE co_gt > {co[0]} AND co_gt < {co[1]}"
            rows = session.execute(query)
            df = pd.DataFrame(rows)
            st.dataframe(df)
        if st.button("Reset", type="primary"):
            pass
    if selected == "Giá trị cố định":
        st.subheader("Lọc dữ liệu theo giá trị CO(GT)")
        co = st.text_input("Chọn giá trị CO(GT)", value="0.1")
        co = float(co)
        if st.button("Lọc dữ liệu"):
            if co >= 0:
                #st.subheader(f"Du lieu co gia tri CO(GT) la {co}")
                query = f"SELECT * FROM air_quality WHERE co_gt = {co}"
                rows = session.execute(query)
                df = pd.DataFrame(rows)
                st.dataframe(df)
            else:
                st.error("Giá trị CO không thể bé hơn 0")
        if st.button("Reset", type="primary"):
            pass