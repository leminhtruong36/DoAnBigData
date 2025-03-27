import streamlit as st
import pandas as pd
import joblib

# 📌 Load mô hình đã train
model = joblib.load("rf_model.pkl")  # Đảm bảo bạn đã lưu model này

# 📌 Tiêu đề
st.title("🔍 Dự đoán chất lượng không khí")

# 📌 Input từ người dùng
co = st.number_input("Nhập nồng độ CO (mg/m³)", min_value=0.0, step=0.1)
no2 = st.number_input("Nhập nồng độ NO₂ (ppb)", min_value=0.0, step=0.1)
nox = st.number_input("Nhập nồng độ NOx (ppb)", min_value=0.0, step=0.1)
ah = st.number_input("Nhập độ ẩm tuyệt đối (g/m³)", min_value=0.0, step=0.1)
c6h6 = st.number_input("Nhập nồng độ C6H6 (µg/m³)", min_value=0.0, step=0.1)
nmhc = st.number_input("Nhập nồng độ NMHC (µg/m³)", min_value=0.0, step=0.1)
pt08_s1_co = st.number_input("Nhập giá trị PT08.S1(CO)", min_value=0.0, step=0.1)
pt08_s2_nmhc = st.number_input("Nhập giá trị PT08.S2(NMHC)", min_value=0.0, step=0.1)
pt08_s3_nox = st.number_input("Nhập giá trị PT08.S3(NOx)", min_value=0.0, step=0.1)
pt08_s4_no2 = st.number_input("Nhập giá trị PT08.S4(NO2)", min_value=0.0, step=0.1)
pt08_s5_o3 = st.number_input("Nhập giá trị PT08.S5(O3)", min_value=0.0, step=0.1)
rh = st.number_input("Nhập độ ẩm tương đối (%)", min_value=0.0, step=0.1)
t = st.number_input("Nhập nhiệt độ (°C)", min_value=0.0, step=0.1)

# 📌 Nút dự đoán
if st.button("Dự đoán"):
    # Tạo DataFrame từ input
    input_data = pd.DataFrame([[co, no2, nox, ah, c6h6, nmhc, pt08_s1_co, pt08_s2_nmhc, pt08_s3_nox, pt08_s4_no2, pt08_s5_o3, rh, t]],
                              columns=["co_gt", "no2_gt", "nox_gt", "ah", "c6h6_gt", "nmhc_gt", "pt08_s1_co", "pt08_s2_nmhc", "pt08_s3_nox", "pt08_s4_no2", "pt08_s5_o3", "rh", "t"])
    
    # Thực hiện dự đoán
    prediction = model.predict(input_data)[0]
    
    # Hiển thị kết quả
    result = "🌍 Có ô nhiễm" if prediction == 1 else "✅ Không ô nhiễm"
    st.success(result)