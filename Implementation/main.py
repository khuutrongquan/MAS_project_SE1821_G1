import pandas as pd
import numpy as np
from scipy.stats import poisson

# Đọc dữ liệu
df = pd.read_csv("creditcard.csv")

# Chuyển đổi thời gian từ giây sang phút
df["Minute"] = df["Time"] // 60

# Áp dụng Log Transformation
df["Log_Amount"] = np.log1p(df["Amount"])

# Tính Z-score
df["Z_Score"] = (df["Log_Amount"] - df["Log_Amount"].mean()) / df["Log_Amount"].std()

# Đếm số lượng giao dịch trong mỗi phút
transaction_counts = df["Minute"].value_counts().sort_index()

# Tính lambda (trung bình số giao dịch mỗi phút)
lambda_poisson = transaction_counts.mean()

# Điều chỉnh ngưỡng Poisson
pmf_threshold = 0.001
cdf_threshold = 0.02

def detect_anomaly_pmf(k, lambda_poisson, pmf_threshold):
    """Xác định bất thường dựa trên PMF"""
    pmf_prob = poisson.pmf(k, lambda_poisson)
    return pmf_prob < pmf_threshold

def detect_anomaly_cdf(k, lambda_poisson, cdf_threshold):
    """Xác định bất thường dựa trên CDF"""
    low_prob = poisson.cdf(k, lambda_poisson)
    high_prob = 1 - poisson.cdf(k, lambda_poisson)
    return (low_prob < cdf_threshold) or (high_prob < cdf_threshold)

# Áp dụng cho từng phút
pmf_anomalous_minutes = [minute for minute, count in transaction_counts.items() if detect_anomaly_pmf(count, lambda_poisson, pmf_threshold)]
cdf_anomalous_minutes = [minute for minute, count in transaction_counts.items() if detect_anomaly_cdf(count, lambda_poisson, cdf_threshold)]

# Gán nhãn bất thường theo Poisson
df["Poisson_PMF_Anomaly"] = df["Minute"].isin(pmf_anomalous_minutes).astype(int)
df["Poisson_CDF_Anomaly"] = df["Minute"].isin(cdf_anomalous_minutes).astype(int)

# Điều chỉnh ngưỡng IQR cho Amount
q1 = df["Amount"].quantile(0.25)
q3 = df["Amount"].quantile(0.75)
iqr = q3 - q1
low_amount = q1 - 3 * iqr
high_amount = q3 + 3 * iqr

# Xác định bất thường theo Amount
df["IQR_Amount_Anomaly"] = ((df["Amount"] < low_amount) | (df["Amount"] > high_amount)).astype(int)

# Ngưỡng anomaly: Z-score > 3 hoặc < -3
z_threshold = 3
df["Gaussian_Anomaly"] = ((df["Z_Score"] > z_threshold) | (df["Z_Score"] < -z_threshold)).astype(int)

# # Xóa cột không cần thiết trước khi lưu
# df.drop(columns=["Minute", "Z_Score"], inplace=True)

# Lưu kết quả
df.to_csv("creditcard_final_anomalies.csv", index=False)

print("File creditcard_final_anomalies.csv đã được tạo với các cột:")
print("- Poisson_PMF_Anomaly: 1 nếu số lượng giao dịch trong phút đó bất thường theo PMF.")
print("- Poisson_CDF_Anomaly: 1 nếu số lượng giao dịch trong phút đó bất thường theo CDF.")
print("- IQR_Amount_Anomaly: 1 nếu số tiền giao dịch quá cao hoặc quá thấp.")
print("- Gaussian_Anomaly: 1 nếu Amount bất thường theo Gaussian Z-score.")