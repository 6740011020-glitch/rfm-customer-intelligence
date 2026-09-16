import os
import sqlite3
import datetime as dt
import pandas as pd
import urllib.request
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

print("=== 1. กำลังโหลดข้อมูลเข้าสู่ระบบ ===")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
filename = "Online_Retail.xlsx"

if not os.path.exists(filename):
    print("กำลังดาวน์โหลด Dataset...")
    urllib.request.urlretrieve(url, filename)

raw_df = pd.read_excel(filename)

print("=== 2. ทำ Data Cleaning & Transformation ===")
df = raw_df.dropna(subset=['CustomerID']).copy()
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalSum'] = df['Quantity'] * df['UnitPrice']

# สร้างเชื่อมต่อ Database SQLite (ไฟล์ retail_dw.db จะถูกสร้างให้อัตโนมัติ)
conn = sqlite3.connect('retail_dw.db')
df.to_sql('fact_transactions', conn, if_exists='replace', index=False)

# คำนวณค่า RFM
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSum': 'sum'
}).reset_index()

rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSum': 'Monetary'}, inplace=True)

print("=== 3. ประมวลผล K-Means & Dynamic Labeling ===")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

cluster_stats = rfm.groupby('Cluster').agg({'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'})
vip_cluster = cluster_stats['Monetary'].idxmax()
at_risk_cluster = cluster_stats['Recency'].idxmax()
remaining = [c for c in range(4) if c not in [vip_cluster, at_risk_cluster]]

if len(remaining) >= 2:
    if cluster_stats.loc[remaining[0], 'Recency'] < cluster_stats.loc[remaining[1], 'Recency']:
        new_cluster, general_cluster = remaining[0], remaining[1]
    else:
        new_cluster, general_cluster = remaining[1], remaining[0]
else:
    new_cluster, general_cluster = remaining[0], 0

segment_names = {
    vip_cluster: 'ลูกค้า VIP (High Value)',
    at_risk_cluster: 'ลูกค้าที่เสี่ยงจะสูญเสีย (At-Risk)',
    new_cluster: 'ลูกค้าใหม่ (New)',
    general_cluster: 'ลูกค้าทั่วไป (General)'
}
rfm['Customer_Segment'] = rfm['Cluster'].map(segment_names)

# บันทึกตาราง RFM ลง SQLite
rfm.to_sql('dim_customer_rfm', conn, if_exists='replace', index=False)

print("=== 4. เทรนและบันทึก Machine Learning Model ===")
X = rfm_scaled
y = rfm['Customer_Segment']
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

# เซฟโมเดลและตัวปรับสเกลไว้ในไฟล์ .joblib
joblib.dump(model, 'rfm_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
conn.close()

print("✅ ดำเนินการสร้าง Database และเซฟโมเดลสำเร็จ!")
