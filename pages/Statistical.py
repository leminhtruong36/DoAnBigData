import streamlit as st
import plotly.express as px
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
import cassandra.util

cloud_config= {
  'secure_connect_bundle': 'secure-connect-doanbigdata.zip'
}
# This token JSON file is autogenerated when you download your token, 
# if yours is different update the file name below

astra_token = st.secrets["ASTRA_DB_TOKEN"]
astra_token_dict = json.loads(astra_token)

CLIENT_ID = astra_token_dict["clientId"]
CLIENT_SECRET = astra_token_dict["secret"]

auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
session.set_keyspace('final')

rows = session.execute("SELECT * FROM air_quality")
data = pd.DataFrame(rows, columns=["date", "time", "co", "pt08_s1_co", "nmhc_gt", "c6h6_gt", "pt08_s2_nmhc", 
                                   "nox_gt", "pt08_s3_nox", "no2_gt", "pt08_s4_no2", "pt08_s5_o3", "t", "rh", "ah"])

data["date"] = data["date"].apply(lambda x: x.date() if isinstance(x, cassandra.util.Date) else x)
data["date"] = pd.to_datetime(data["date"])

st.title("📊 Biểu đồ chỉ số ô nhiễm theo tháng")

month = st.selectbox("Chọn tháng", list(range(1, 13)), index=2)
year = st.selectbox("Chọn năm", sorted(data["date"].dt.year.unique()), index=0)

pollutant_mapping = {
    "CO(GT)": "co",
    "NOx(GT)": "nox_gt",
    "NO2(GT)": "no2_gt",
    "NMHC(GT)": "nmhc_gt",
    "C6H6(GT)": "c6h6_gt",
    "T": "t",
    "RH": "rh",
    "AH": "ah"
}

selected_pollutant = st.selectbox("Chọn chỉ số ô nhiễm", list(pollutant_mapping.keys()))

if st.button("📊 Hiển thị biểu đồ"):
    actual_column = pollutant_mapping[selected_pollutant]

    filtered_data = data[(data["date"].dt.month == month) & (data["date"].dt.year == year)]

    if filtered_data.empty:
        st.warning("⚠️ Không có dữ liệu cho tháng này.")
    else:
        daily_avg = filtered_data.groupby("date")[actual_column].mean().reset_index()

        fig = px.line(
            daily_avg, 
            x="date", 
            y=actual_column, 
            title=f"{selected_pollutant} trung bình theo ngày - Tháng {month}/{year}",
            labels={"date": "Ngày", actual_column: selected_pollutant},
            line_shape="spline"
        )
        st.plotly_chart(fig)
