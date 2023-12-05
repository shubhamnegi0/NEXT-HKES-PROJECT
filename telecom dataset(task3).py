#!/usr/bin/env python
# coding: utf-8

# # Task 3 - Experience Analytics

# In[1]:


# Task 3.1 - Aggregate, per customer, the following information:


# In[2]:


# • Average TCP retransmission

# • Average RTT

# • Handset type

# • Average throughput


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv("telcom_data.csv")


# In[5]:


def new_columns(df):
    df.columns= [column.replace(' ', '_').lower() for column in df.columns]
    return df


# In[10]:


df = new_columns(df)


# In[11]:


df1 =df.rename(columns={'msisdn/number': 'msisdn', 'dur._(ms)': 'duration' })


# In[12]:


# We need to combine the total UL and Dl data and Create a new DataFrame


# In[13]:


df1["tcp_retrans"] = df1["tcp_dl_retrans._vol_(bytes)"] + df1['tcp_ul_retrans._vol_(bytes)']
df1["avg_rtt"] = df1["avg_rtt_dl_(ms)"] + df1["avg_rtt_ul_(ms)"]
df1['avg_tp'] = df1["avg_bearer_tp_dl_(kbps)"] + df1["avg_bearer_tp_ul_(kbps)"]


# In[14]:


# Replace missing values and outliers with the mean or mode of the corresponding variable


# In[15]:


df1.isnull().sum()


# In[17]:


df1['tcp_retrans'].fillna(df1['tcp_retrans'].mean(), inplace=True)

df1['avg_rtt'].fillna(df1['avg_rtt'].mean(), inplace=True)

df1['avg_tp'].fillna(df1['avg_tp'].mean(), inplace=True)

df1['handset_type'].fillna(df1['handset_type'].mode()[0], inplace=True)


# In[20]:


# df1.isnull().sum()


# # Group by customer (msisdn ) and calculate the average of each variable
# agg_df = df1.groupby('msisdn').agg({
#     'tcp_retrans': 'mean',
#     'avg_rtt': 'mean',
#     'handset_type': 'first',
#     'avg_tp': 'mean'
# }).reset_index()
# print(agg_df)

# In[23]:


agg_df.head(10).style.background_gradient(cmap = "Greens")


# # Task 3.2 - Compute & list 10 of the top, bottom and most frequent:
# 

# In[26]:


#  a. TCP values in the dataset.
# b. RTT values in the dataset.
# c. Throughput values in the dataset.


# In[30]:


# Compute the top, bottom, and most frequent values for TCP
top_tcp = df1['tcp_retrans'].nlargest(10)
bottom_tcp = df1['tcp_retrans'].nsmallest(10)
most_frequent_tcp = df1['tcp_retrans'].value_counts().head(10)
most_frequent_tcp


# In[31]:


# Compute the top, bottom, and most frequent values for RTT
top_rtt = df1['avg_rtt'].nlargest(10)
bottom_rtt = df1['avg_rtt'].nsmallest(10)
most_frequent_rtt = df1['avg_rtt'].value_counts().head(10)
most_frequent_rtt


# In[34]:


# Compute the top, bottom, and most frequent values for throughput
top_throughput = df1['avg_tp'].nlargest(10)
bottom_throughput = df1['avg_tp'].nsmallest(10)
most_frequent_throughput = df1['avg_tp'].value_counts().head(10)
most_frequent_throughput


# In[35]:


# Print the results


# In[37]:


# the most_frequent_tcp
print("*"*50)
print("Top TCP values:")
print(top_tcp)
print("\nBottom TCP values:")
print(bottom_tcp)
print("\nMost frequent TCP values:")
print(most_frequent_tcp)
print("*"*50)


# In[40]:


# the Most frequent RTT values
print("*"*50)
print("\nTop RTT values:")
print(top_rtt)
print("\nBottom RTT values:")
print(bottom_rtt)
print("\nMost frequent RTT values:")
print(most_frequent_rtt)
print("*"*50)


# In[42]:


# the most_frequent_throughput
print("*"*50)
print("\nTop throughput values:")
print(top_throughput)
print("\nBottom throughput values:")
print(bottom_throughput)
print("\nMost frequent throughput values:")
print(most_frequent_throughput)
print("*"*50)


# In[43]:


# Create a DataFrame with the top, bottom, and most frequent values


# In[45]:


pairplot_df = pd.DataFrame({
    'Top TCP': top_tcp,
    'Bottom TCP': bottom_tcp,
    'Most Frequent TCP': most_frequent_tcp,
    'Top RTT': top_rtt,
    'Bottom RTT': bottom_rtt,
    'Most Frequent RTT': most_frequent_rtt,
    'Top Throughput': top_throughput,
    'Bottom Throughput': bottom_throughput,
    'Most Frequent Throughput': most_frequent_throughput
})


# In[46]:


# Plot the pair plot
sns.pairplot(pairplot_df)
plt.show()


# # Task 3.3 - Compute & report
# 

# In[48]:


# Compute the distribution of average throughput per handset type
throughput_distribution = df1.groupby('handset_type')['avg_tp'].mean()


# In[49]:


# Compute the average TCP retransmission view per handset type
tcp_retransmission_view = df1.groupby('handset_type')['tcp_retrans'].mean()


# In[50]:


# Print the results
print("Distribution of Average Throughput per Handset Type:")
print(throughput_distribution)
print("\nAverage TCP Retransmission View per Handset Type:")
print(tcp_retransmission_view)


# In[51]:


# Task 3.4 - Using the experience metrics above, perform a k-means clustering (where k = 3) to segment users into groups of experiences and provide a brief description of each cluster. (The description must define each group based on your understanding of the data)


# In[52]:


from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.cluster import KMeans


# In[53]:


from sklearn.preprocessing import StandardScaler
# Normalize the experience  metrics
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df1[['avg_tp', 'tcp_retrans', 'avg_rtt']]), columns=['avg_tp', 'tcp_retrans', 'avg_rtt'], index=df.index)
normalized_df


# In[54]:


# Apply K-means clustering with k=3
kmeans = KMeans(n_clusters=3, init='k-means++')
df1['cluster'] = kmeans.fit_predict(normalized_df)


# In[56]:


# Interpret the clusters
cluster_means = df1.groupby('cluster')[['avg_tp', 'tcp_retrans','avg_rtt']].mean()
print("Cluster Means:")
print(cluster_means)


# In[58]:


experience_metrics_with_cluster = agg_df.copy()
experience_metrics_with_cluster['clusters'] = df1['cluster']
print(experience_metrics_with_cluster['clusters'])


# In[59]:


from mpl_toolkits.mplot3d import Axes3D


# In[60]:


# Apply K-means clustering with k=3
kmeans = KMeans(n_clusters=3, init='k-means++')
df1['cluster'] = kmeans.fit_predict(normalized_df)


# In[61]:


# Interpret the clusters
cluster_means = df1.groupby('cluster')[['avg_tp', 'tcp_retrans', 'avg_rtt']].mean()
print("Cluster Means:")
print(cluster_means)


# In[66]:


# Create a 3D  plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

x = df1['avg_tp']
y = df1['tcp_retrans']
z = df1['avg_rtt']
c = df1['cluster']

scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=20)

ax.set_xlabel('Average Throughput')
ax.set_ylabel('TCP Retransmission')
ax.set_zlabel('Average RTT')

cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')
plt.title('Clusters of User Experience Metrics (3D Scatter Plot)')
plt.show()

experience_metrics_with_cluster = agg_df.copy()
experience_metrics_with_cluster['clusters'] = df1['cluster']


# In[63]:





# In[64]:





# In[ ]:




