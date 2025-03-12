import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import precision_score, recall_score, f1_score

# Đọc dữ liệu
file_path = "creditcard.csv"  # Đổi đường dẫn nếu cần
df = pd.read_csv(file_path)

# Chuyển đổi thời gian từ giây sang phút
df["Minute"] = df["Time"] // 60

# Đếm số lượng giao dịch trong mỗi phút
transaction_counts = df["Minute"].value_counts().sort_index()

# Tính lambda (trung bình số giao dịch mỗi phút)
lambda_poisson = transaction_counts.mean()

# Nhãn thực tế (ground truth)
y_true = df["Class"]  # Class = 1 là gian lận, 0 là bình thường

# Tạo danh sách các ngưỡng thử nghiệm
pmf_thresholds = [0.001, 0.005, 0.01]
cdf_thresholds = [0.01, 0.02, 0.05]

# Lưu kết quả
results = []

for pmf_threshold in pmf_thresholds:
    for cdf_threshold in cdf_thresholds:
        # Xác định bất thường theo PMF
        pmf_anomalous_minutes = [minute for minute, count in transaction_counts.items() 
                                 if poisson.pmf(count, lambda_poisson) < pmf_threshold]
        
        # Xác định bất thường theo CDF
        cdf_anomalous_minutes = [minute for minute, count in transaction_counts.items() 
                                 if (poisson.cdf(count, lambda_poisson) < cdf_threshold) or 
                                    (1 - poisson.cdf(count, lambda_poisson) < cdf_threshold)]
        
        # Gán nhãn bất thường
        df["Poisson_PMF_Anomaly"] = df["Minute"].isin(pmf_anomalous_minutes).astype(int)
        df["Poisson_CDF_Anomaly"] = df["Minute"].isin(cdf_anomalous_minutes).astype(int)

        # Xác định giao dịch là gian lận nếu bị đánh dấu bởi bất kỳ phương pháp nào
        y_pred = (df["Poisson_PMF_Anomaly"] | df["Poisson_CDF_Anomaly"]).astype(int)

        # Tính toán precision, recall, f1-score
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Lưu kết quả
        results.append({
            "PMF Threshold": pmf_threshold,
            "CDF Threshold": cdf_threshold,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results)

# Hiển thị kết quả
print(results_df)

# Lưu kết quả ra file CSV
results_df.to_csv("threshold_optimization_results.csv", index=False)
print("Kết quả đã được lưu vào 'threshold_optimization_results.csv'")
