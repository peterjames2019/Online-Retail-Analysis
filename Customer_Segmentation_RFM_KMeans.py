# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:50:53 2026

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset
dataset = pd.read_csv('OnlineRetail.csv', encoding="ISO-8859-1")

# 2. Remove rows without a CustomerID, CustomerID is very important for grouping
dataset = dataset.dropna(subset=['CustomerID'])

# 3. Filter out 'Cancelled' orders (Invoices starting with 'C')
dataset = dataset[~dataset['InvoiceNo'].str.contains('C', na=False)]

# 4. Create a column named 'Total Sum' for  (Quantity * UnitPrice)
dataset['TotalSum'] = dataset['Quantity'] * dataset['UnitPrice']

# 5. Set the "Snapshot Date" to one day after the last transaction (hypothetical today)
import datetime as dt

# 5a. first of all Convert the InvoiceDate column to actual datetime objects
dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'])

#5b. Now let'snapshot it
snapshot_date = dataset['InvoiceDate'].max() + dt.timedelta(days=1)

# 6. Calculate RFM Values
rfm_dataset = dataset.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'count',                                   # Frequency
    'TotalSum': 'sum'                                       # Monetary
})

# Rename columns for clarity
rfm_dataset.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalSum': 'MonetaryValue'}, inplace=True)

#We need to scale since we are going to use kmeans clustering model

from sklearn.preprocessing import StandardScaler

# 1. Handle zeros or negative values (Log only works on positive numbers)
# We add a small constant (1) just in case there are 0s
rfm_log = np.log(rfm_dataset + 1)

# 2. Initialize the Scaler
scaler = StandardScaler()

# 3. Fit and Transform the data
rfm_scaled = scaler.fit_transform(rfm_log)

# 4. Turn it back into a readable dataset (Optional but helpful)
rfm_scaled = pd.DataFrame(rfm_scaled, index=rfm_dataset.index, columns=rfm_dataset.columns)

print(rfm_scaled.head())


#Now LET"S FEED it to our Kmeans Model
from sklearn.cluster import KMeans

# 1. Initialize the model
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)

# 2. Fit and predict
rfm_dataset['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 3. Check the first few rows
print(rfm_dataset.head())


#ANALYSING THE CLUSTER VALUES
# Calculate average values for each cluster
cluster_analysis = rfm_dataset.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count'] # 'count' tells us how many customers are in each group
}).round(1)

print(cluster_analysis)

# Create a dictionary to map numbers to names
segment_map = {
    1: 'Champions',
    3: 'Loyal Customers',
    0: 'New/Recent Customers',
    2: 'Hibernating'
}

# Apply the map to the Cluster column
rfm_dataset['Segment'] = rfm_dataset['Cluster'].map(segment_map)

# Check the results
print(rfm_dataset[['Recency', 'Frequency', 'MonetaryValue', 'Segment']].head())


#Visualizing the Result
import squarify 

# 1. Calculate the number of customers in each segment
segment_counts = rfm_dataset['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']

# 2. Plotting the Treemap
plt.figure(figsize=(12, 8))
squarify.plot(sizes=segment_counts['Count'], 
              label=segment_counts['Segment'], 
              alpha=0.8, 
              color=sns.color_palette("Spectral", len(segment_counts)))

plt.title("Customer Segmentation Treemap", fontsize=18)
plt.axis('off')
plt.show()