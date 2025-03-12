from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
# Đọc dữ liệu đã lưu (nếu cần thiết)
df = pd.read_csv("creditcard_final_anomalies.csv")

# Tính các chỉ số cho Poisson PMF
accuracy_pmf = accuracy_score(df['Class'], df['Poisson_PMF_Anomaly'])
precision_pmf = precision_score(df['Class'], df['Poisson_PMF_Anomaly'])
recall_pmf = recall_score(df['Class'], df['Poisson_PMF_Anomaly'])
f1_pmf = f1_score(df['Class'], df['Poisson_PMF_Anomaly'])

# Tính các chỉ số cho Poisson CDF
accuracy_cdf = accuracy_score(df['Class'], df['Poisson_CDF_Anomaly'])
precision_cdf = precision_score(df['Class'], df['Poisson_CDF_Anomaly'])
recall_cdf = recall_score(df['Class'], df['Poisson_CDF_Anomaly'])
f1_cdf = f1_score(df['Class'], df['Poisson_CDF_Anomaly'])

# Tính các chỉ số cho IQR Amount
accuracy_iqr = accuracy_score(df['Class'], df['IQR_Amount_Anomaly'])
precision_iqr = precision_score(df['Class'], df['IQR_Amount_Anomaly'])
recall_iqr = recall_score(df['Class'], df['IQR_Amount_Anomaly'])
f1_iqr = f1_score(df['Class'], df['IQR_Amount_Anomaly'])

# Tính các chỉ số cho Z-Score
accuracy_gaussian = accuracy_score(df['Class'], df['Gaussian_Anomaly'])
precision_gaussian = precision_score(df['Class'], df['Gaussian_Anomaly'])
recall_gaussian = recall_score(df['Class'], df['Gaussian_Anomaly'])
f1_gaussian = f1_score(df['Class'], df['Gaussian_Anomaly'])

# In kết quả cho từng phương pháp
print("Poisson PMF:")
print(f"Accuracy: {accuracy_pmf:.4f}")
print(f"Precision: {precision_pmf:.4f}")
print(f"Recall: {recall_pmf:.4f}")
print(f"F1-score: {f1_pmf:.4f}")
print()

print("Poisson CDF:")
print(f"Accuracy: {accuracy_cdf:.4f}")
print(f"Precision: {precision_cdf:.4f}")
print(f"Recall: {recall_cdf:.4f}")
print(f"F1-score: {f1_cdf:.4f}")
print()

print("IQR Amount:")
print(f"Accuracy: {accuracy_iqr:.4f}")
print(f"Precision: {precision_iqr:.4f}")
print(f"Recall: {recall_iqr:.4f}")
print(f"F1-score: {f1_iqr:.4f}")
print()

print("Gaussian (Z-score):")
print(f"Accuracy: {accuracy_gaussian:.4f}")
print(f"Precision: {precision_gaussian:.4f}")
print(f"Recall: {recall_gaussian:.4f}")
print(f"F1-score: {f1_gaussian:.4f}")
