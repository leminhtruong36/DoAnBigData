import streamlit as st
import pandas as pd
import joblib
import requests
import os

# 📌 URL GitHub chứa model (CẬP NHẬT URL CỦA BẠN)
GITHUB_BASE_URL = "https://raw.githubusercontent.com/leminhtruong36/DoAnBigData/main/"
MODEL_PATH ="https://raw.githubusercontent.com/leminhtruong36/DoAnBigData/main/rf_model.pkl"
MODEL_URL = "rf_model.pkl"
METRICS_URL = GITHUB_BASE_URL + "rf_metrics.json"
TEST_SAMPLES_URL = GITHUB_BASE_URL + "test_samples.json"
TEST_RESULTS_URL = GITHUB_BASE_URL + "test_results.csv"

# 📌 Tải model nếu chưa có
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("🔄 Đang tải mô hình dự đoán...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Model tải thành công!")
        else:
            st.error("⚠️ Không thể tải model từ GitHub!")

# 📌 Kiểm tra và tải model
download_model()

# 📌 Load mô hình
try:
    model = joblib.load(MODEL_PATH)
    st.success("🚀 Mô hình đã sẵn sàng!")
except Exception as e:
    st.error(f"❌ Lỗi tải model: {e}")

st.title("📊 Kết quả huấn luyện mô hình Random Forest")
# 📌 Tải và hiển thị metrics
try:
    response = requests.get(METRICS_URL)
    response.raise_for_status()
    metrics = response.json()

    st.write("### 🔥 Đánh giá mô hình")
    st.metric("🎯 Accuracy", f"{metrics['accuracy']:.4f}")
    st.metric("🎯 Precision", f"{metrics['precision']:.4f}")
    st.metric("🎯 Recall", f"{metrics['recall']:.4f}")
    st.metric("🎯 F1-score", f"{metrics['f1_score']:.4f}")
except Exception as e:
    st.error(f"⚠️ Không thể tải file metrics: {e}")

# 📌 Hiển thị một số mẫu test
st.write("### 📌 Một số bản ghi từ tập test")
try:
    response = requests.get(TEST_SAMPLES_URL)
    response.raise_for_status()
    test_samples = response.json()
    df_samples = pd.DataFrame(test_samples)
    st.dataframe(df_samples)
except Exception as e:
    st.error(f"⚠️ Không thể tải file test_samples: {e}")

# 📌 Cho phép người dùng tải toàn bộ tập test
st.write("### 📥 Tải toàn bộ kết quả test")
try:
    response = requests.get(TEST_RESULTS_URL)
    response.raise_for_status()
    st.download_button(
        label="📥 Tải toàn bộ kết quả test",
        data=response.content,
        file_name="test_results.csv",
        mime="text/csv"
    )
except Exception as e:
    st.error(f"⚠️ Không thể tải file test_results: {e}")



# 📌 Tiêu đề ứng dụng
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
    if "model" in locals():
        # Tạo DataFrame từ input
        input_data = pd.DataFrame([[co, no2, nox, ah, c6h6, nmhc, pt08_s1_co, pt08_s2_nmhc, pt08_s3_nox, pt08_s4_no2, pt08_s5_o3, rh, t]],
                                  columns=["co_gt", "no2_gt", "nox_gt", "ah", "c6h6_gt", "nmhc_gt", "pt08_s1_co", "pt08_s2_nmhc", "pt08_s3_nox", "pt08_s4_no2", "pt08_s5_o3", "rh", "t"])
        
        # Thực hiện dự đoán
        prediction = model.predict(input_data)[0]
        
        # Hiển thị kết quả
        result = "🌍 Có ô nhiễm" if prediction == 1 else "✅ Không ô nhiễm"
        st.success(result)
    else:
        st.error("⚠️ Model chưa được tải thành công, không thể dự đoán!")

