import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("Walmart_customer_purchases_cleaned.csv")

# Convert date
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
reference_date = df['Purchase_Date'].max() + pd.Timedelta(days=1)

# Create RFM table
rfm = df.groupby('Customer_ID').agg({
    'Purchase_Date': lambda x: (reference_date - x.max()).days,
    'Customer_ID': 'count',
    'Purchase_Amount': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Normalize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Save clustered output
rfm.to_csv("Customer_Segments.csv")
print("Segmentation saved as 'Customer_Segments.csv'")


import seaborn as sns

sns.pairplot(rfm.reset_index(), hue='Cluster', vars=['Recency', 'Frequency', 'Monetary'])
plt.show()



# Load clustered data
rfm = pd.read_csv("Customer_Segments.csv")

# Bar chart for customer count per segment
plt.figure(figsize=(8, 5))
sns.countplot(x='Cluster', data=rfm, palette='Set2')
plt.title('Customer Count per Segment')
plt.xlabel('Customer Segment (Cluster)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()
