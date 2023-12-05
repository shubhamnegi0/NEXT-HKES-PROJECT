#!/usr/bin/env python
# coding: utf-8

# # Task 2 : User Engagement Analysis
# 

# In[83]:


# sessions frequency 
# the duration of the session 
# the session total traffic (download and upload (bytes))


# In[13]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[15]:


df = pd.read_csv("telcom_data.csv")
df.head()


# In[17]:


#Convert spaces into underscore in column name of dataframe and Uppercase to lowercase


# In[18]:


def new_columns(df):
    df.columns= [column.replace(' ', '_').lower() for column in df.columns]
    return df


# In[20]:


df = new_columns(df)
df


# In[22]:


# We need to combine the total UL and Dl data of each application


# In[30]:


df["social_media"] = df["social_media_dl_(bytes)"] + df['social_media_ul_(bytes)']
df["google"] = df["google_dl_(bytes)"] + df["google_ul_(bytes)"]
df['email'] = df["email_dl_(bytes)"] + df["email_ul_(bytes)"]
df['youtube'] = df["youtube_dl_(bytes)"] + df["youtube_ul_(bytes)"]
df['netflix'] = df["netflix_dl_(bytes)"] + df["netflix_ul_(bytes)"]
df["gaming"] = df["gaming_dl_(bytes)"] + df["gaming_ul_(bytes)"]
df['other']= df["other_dl_(bytes)"]+ df["other_ul_(bytes)"]
df['total_data'] = df['total_dl_(bytes)'] + df['total_ul_(bytes)']

df["social_media"]


# In[26]:


# df.info()


# In[28]:


df1 =df.rename(columns={'msisdn/number': 'msisdn', 'dur._(ms)': 'duration' })
df1


# In[31]:


#Aggregate the sessions frequency count for bearer id


# In[36]:


session_count = df1.groupby('msisdn').agg({'bearer_id':'count'})
session_freq_metrics = session_count.sort_values(by=['bearer_id'],ascending=False)
session_freq_metrics.head(10)


# In[37]:


#Aggregate the sessions frequency count for duration


# In[38]:


duration_count = df1.groupby('msisdn').agg({'duration':'sum'})
duration_metrics = duration_count.sort_values(by=['duration'],ascending=False)
duration_metrics.head(10)


# In[39]:


#Aggregate the sessions frequency for total data


# In[40]:


total_data_count = df1.groupby('msisdn').agg({'total_data':'sum'})
total_data_metrics = total_data_count.sort_values(by=['total_data'],ascending=False)
total_data_metrics.head(10)


#  # Task 2.1
# 

# In[42]:


# Aggregate the above metrics per customer id (MSISDN) and report the top 10 customers per engagement metric


# In[41]:


engagement_metrics = df1.groupby('msisdn').agg({'bearer_id': 'count','duration': 'sum', 'total_data': 'sum',})
engagement_metrics = engagement_metrics.rename(columns={'bearer_id': 'session_frequency', 'total_data': 'total_traffic'})
engagement_metrics.sort_values(by=['session_frequency'], ascending=False)


# In[43]:


engagement_metrics.describe().style.background_gradient(cmap = "Blues")


# In[44]:


sns.boxplot(data=engagement_metrics, x='session_frequency').set(title="Outlier detection for bearer_id count")


# In[45]:


sns.boxplot(data=engagement_metrics, x='duration').set(title="Outlier detection for Total duration")


# In[46]:


sns.boxplot(data=engagement_metrics, x='total_traffic').set(title="Outlier detection for Total_data")


# In[47]:


# There are outliers present in above three metrics.


# In[48]:


def handle_outliers(df1, col):
    df1 = df1.copy()
    q1 = df1[col].quantile(0.25)
    q2 = df1[col].quantile(0.50)
    q3 = df1[col].quantile(0.75)
    
    iqr=q3-q1
    lower_range=q1-iqr*1.5
    upper_range=q3+iqr*1.5
    
    df1[col] = np.where(df1[col] < lower_range, lower_range, df1[col])
    df1[col] = np.where(df1[col] > upper_range, upper_range, df1[col])
    
    return df1


# In[49]:


new_engagement_metrics = handle_outliers(engagement_metrics, 'session_frequency')
new_engagement_metrics = handle_outliers(new_engagement_metrics, 'duration')
new_engagement_metrics = handle_outliers(new_engagement_metrics, 'total_traffic')


# In[50]:


new_engagement_metrics.describe().style.background_gradient(cmap = "Reds")


# In[51]:


sns.boxplot(data=new_engagement_metrics, x='session_frequency').set(title="Outlier detection for bearer_id count")


# In[52]:


sns.boxplot(data=new_engagement_metrics, x='duration').set(title="Outlier detection for Total duration")


# In[53]:


sns.boxplot(data=new_engagement_metrics, x='total_traffic').set(title="Outlier detection for Total_data")


# In[54]:


# • Plot the top 3 most used applications using appropriate charts.


# In[55]:


pp_metrics = []

app_columns = ['social_media', 'google', 'email', 'youtube', 'netflix', 'gaming']

app_metrics = df1.groupby('msisdn').agg({'social_media': 'sum', 'google': 'sum', 'email': 'sum',
                                        'youtube': 'sum', 'netflix': 'sum', 'gaming': 'sum'})



# In[57]:


figure, axes = plt.subplots(3, 2, figsize=(18,14))

count = 0
for app_metric in app_metrics:
    
    x = app_metrics[[app_metric]].sort_values(by = [app_metric],
                                              ascending=False).head(10)
    x.plot.bar(y=app_metric, ax=axes[(count//2), count%2], rot=12)
    count += 1


# In[58]:


total_app_df = pd.DataFrame(columns=['app', 'total'])
total_app_df['app'] = app_columns

app_metrics
apps_total = []

for app in app_columns:
    apps_total.append(app_metrics.sum()[app])
total_app_df['total'] =apps_total
    
total_app_df


# In[59]:


#Plot the results


# In[60]:


plt.figure(figsize=(10,8))
sns.barplot(data=total_app_df, x='app', y='total')
plt.xlabel("Application")
plt.ylabel("Total data volume")
plt.title("each application total usage data")
plt.show()


# In[ ]:


# Normalize each engagement metric and run a k-means (k=3) to classify customers into three groups of engagement.


# In[61]:


from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.cluster import KMeans


# In[62]:


metrics_to_normalize = new_engagement_metrics[['session_frequency', 'duration', 'total_traffic']]  
scaler = MinMaxScaler()

norm_metrics = scaler.fit_transform(metrics_to_normalize)
normalized_metrics = pd.DataFrame(norm_metrics, columns=metrics_to_normalize.columns)
print(normalized_metrics)


# In[63]:


normalized_metrics.describe().style.background_gradient(cmap = "Greens")


# In[64]:


sns.displot(data=normalized_metrics, x='session_frequency',kde=True).set(title="Distribution of Session_frequency")


# In[65]:


sns.displot(data=normalized_metrics, x='duration',kde=True).set(title="Distribution of Duration")


# In[66]:


sns.displot(data=normalized_metrics, x='total_traffic',kde=True).set(title="Distribution of Session_frequency")


# In[67]:


#Define a range of k values to evaluate


# In[68]:


k_values = range(1, 20)


# In[69]:


#Initialize a list to store the inertia values
inertia_values = []

#Iterate over each value of k
for k in k_values:
    kmeans = KMeans(n_clusters=k, max_iter=1000)
    kmeans.fit(normalized_metrics)
    inertia_values.append(kmeans.inertia_)


# In[71]:


#Plot the results
plt.figure(figsize=(10,6))
plt.plot(k_values, inertia_values,'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances (inertia)')
plt.title('Elbow Method for Optimal k')
plt.grid()


# In[72]:


# From the results of the above method, we select :
# 
# k : Clusters = 3


# In[73]:


kmeans = KMeans(n_clusters=3, init='k-means++').fit(normalized_metrics)
pred = kmeans.predict(normalized_metrics)

engagement_metrics_with_cluster = new_engagement_metrics.copy()
engagement_metrics_with_cluster['clusters'] = pred


# In[74]:


print(engagement_metrics_with_cluster['clusters'])


# In[75]:


def cluster_label(x: int, cols: list = []) -> str:
    if (not type(x) == int):
        return x
    if (x < len(cols)):
        return cols[x]

    return str(x)


# In[76]:


labels = ['cluster_1', 'cluster_2', 'cluster_3']

engagement_metrics_with_cluster['clusters'] = engagement_metrics_with_cluster['clusters'].map(lambda x: cluster_label(x, labels))

# Now, let's count the occurrences of each cluster label
engagement_metrics_with_cluster['clusters'].value_counts()

plt.figure(figsize=(12,9))

sns.scatterplot(x = 'duration', y='total_traffic', hue='clusters',
                data=engagement_metrics_with_cluster).set_title('Clusters of user engagement metric')
plt.show()

engagement_metrics_with_cluster.describe()


# In[77]:


cluster_1 =  engagement_metrics_with_cluster[engagement_metrics_with_cluster['clusters'] == 'cluster_1']
cluster_2 =  engagement_metrics_with_cluster[engagement_metrics_with_cluster['clusters'] == 'cluster_2']
cluster_3 =  engagement_metrics_with_cluster[engagement_metrics_with_cluster['clusters'] == 'cluster_3']


# In[79]:


cluster_1.describe()
cluster_2.describe()
cluster_3.describe()



# In[ ]:




