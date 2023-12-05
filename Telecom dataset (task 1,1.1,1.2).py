#!/usr/bin/env python
# coding: utf-8

# # Import all the nesscary liabries

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[28]:


df = pd.read_csv("telcom_data.csv",na_values=["?",None])
df


# In[3]:


df.head()


# # Exploratory data analysis

# In[29]:


df.describe()


# In[30]:


df.info()


# In[6]:


#to check the number of null values in columns
df.isnull().sum()


# In[7]:


df.columns.to_list()


# In[31]:


print(f" there are {df.shape[0]} rows and {df.shape[1]} columns")


# In[9]:


df.columns


# # Task 1 - User Overview Analysis 

# # identify the Top 10 handset used by customers

# In[32]:


headset_count = df['Handset Type'].value_counts()
top_10_handsets = headset_count.head(10)
print(top_10_handsets)


# # identify the top 3 handset manufacturers

# In[33]:


manufactures_count = df ["Handset Manufacturer"].value_counts()
top_3_manufactures = manufactures_count.head(3)
print(top_3_manufactures)


# # identify the top 5 handsets per top 3 handset manufacturer

# In[34]:


top_handset = df.groupby("Handset Manufacturer")["Handset Type"].apply(lambda x: x.head(5)).reset_index(drop = True)
print(top_handset)


# In[13]:


df_clean = df.drop(["TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)",
                    "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)",
                    "HTTP DL (Bytes)","HTTP UL (Bytes)",
                    'Avg RTT DL (ms)','Avg RTT UL (ms)','Nb of sec with 1250B < Vol UL < 6250B',
       'Nb of sec with 31250B < Vol DL < 125000B',
       'Nb of sec with 37500B < Vol UL',
       'Nb of sec with 6250B < Vol DL < 31250B',
       'Nb of sec with 6250B < Vol UL < 37500B'],axis =1)
df_clean.shape
# df_clean.isnull().sum()


# In[35]:


#fill missing values with fill method for
def fix_missing_ffill (df,col):
    df[col] = df[col].fillna(method = "ffill")
    return df[col]

def fix_missing_bfill(df,col):
    df[col] = df[col].fillna(method = "bfill")
    return df[col]


# In[15]:


#need to perform fillna for all the columns which have null values

df_clean["Bearer Id"] = fix_missing_ffill(df_clean,"Bearer Id")

df_clean["IMSI"] = fix_missing_ffill(df_clean,"IMSI")

df_clean["MSISDN/Number"] = fix_missing_ffill(df_clean,"MSISDN/Number")

df_clean["IMEI"] = fix_missing_ffill(df_clean,"IMEI")

df_clean["Last Location Name"] = fix_missing_ffill(df_clean,"Last Location Name")

df_clean["DL TP < 50 Kbps (%)"] = fix_missing_ffill(df_clean,"DL TP < 50 Kbps (%)")

df_clean["50 Kbps < DL TP < 250 Kbps (%)"] = fix_missing_ffill(df_clean,"50 Kbps < DL TP < 250 Kbps (%)")

df_clean["250 Kbps < DL TP < 1 Mbps (%)"] = fix_missing_ffill(df_clean,"250 Kbps < DL TP < 1 Mbps (%)")

df_clean["DL TP > 1 Mbps (%)"] = fix_missing_ffill(df_clean,"DL TP > 1 Mbps (%)")

df_clean["UL TP < 10 Kbps (%)"] = fix_missing_ffill(df_clean,"UL TP < 10 Kbps (%)")

df_clean["UL TP < 10 Kbps (%)"] = fix_missing_ffill(df_clean,"UL TP < 10 Kbps (%)")

df_clean["50 Kbps < UL TP < 300 Kbps (%)"] = fix_missing_ffill(df_clean,"50 Kbps < UL TP < 300 Kbps (%)")

df_clean["UL TP > 300 Kbps (%)"] = fix_missing_ffill(df_clean,"UL TP > 300 Kbps (%)")

df_clean["Handset Manufacturer"] = fix_missing_ffill(df_clean,"Handset Manufacturer")

df_clean["Handset Type"] = fix_missing_ffill(df_clean,"Handset Type")

df_clean["Nb of sec with Vol DL < 6250B"] = fix_missing_ffill(df_clean,"Nb of sec with Vol DL < 6250B")

df_clean["Nb of sec with Vol UL < 1250B"] = fix_missing_ffill(df_clean,"Nb of sec with Vol UL < 1250B")

df_clean["10 Kbps < UL TP < 50 Kbps (%)"] = fix_missing_ffill(df_clean,"10 Kbps < UL TP < 50 Kbps (%)")






# In[36]:


# fill the Nb of sec with 125000B < Vol DL values with mode
df_clean["Nb of sec with 125000B < Vol DL"] = df_clean["Nb of sec with 125000B < Vol DL"].fillna(df_clean["Nb of sec with 125000B < Vol DL"].mode()[0])

# df_clean.isnull().sum()




# # Transforming the data 

# Scaling and Normalization
# 
# 

# In[37]:


# Scaling 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

original_data = pd.DataFrame(np.random.exponential(200,size =2000))
original_data.sample(10)


# In[18]:


# is used to check the min and max value 
original_data[0].min(),original_data[0].max()


# In[38]:


counts,bins,ignored = plt.hist(original_data,14)
plt.show()


# # Task 1.1

# # Aggregate per user the following information in the column  

# # number of xDR sessions
# 

# In[39]:


d={"Xdr_Session":df["Bearer Id"],
  "Dur_msec":df["Dur. (ms)"],
  "Activity Duration DL (ms)":df["Activity Duration DL (ms)"],
  "Activity Duration UL (ms)":df["Activity Duration UL (ms)"],
  "Social Media DL (Bytes)":df["Social Media DL (Bytes)"],
  "Social Media UL (Bytes)":df["Social Media UL (Bytes)"],
  "Google DL (Bytes)":df["Google DL (Bytes)"],
  "Google UL (Bytes)":df["Google UL (Bytes)"],
  "Email DL (Bytes)":df["Email DL (Bytes)"],
  "Email UL (Bytes)":df["Email UL (Bytes)"],
  "Youtube DL (Bytes)":df["Youtube DL (Bytes)"],
  "Youtube UL (Bytes)":df["Youtube UL (Bytes)"],
  "Netflix DL (Bytes)":df["Netflix DL (Bytes)"],
  "Netflix UL (Bytes)":df["Netflix UL (Bytes)"],
  "Gaming DL (Bytes)":df["Gaming DL (Bytes)"],
  "Gaming UL (Bytes)":df["Gaming UL (Bytes)"],
  "Other DL (Bytes)":df["Other DL (Bytes)"],
  "Other UL (Bytes)":df["Other UL (Bytes)"],
  "Total UL (Bytes)":df["Total UL (Bytes)"],
  "Total DL (Bytes)":df["Total DL (Bytes)"]}


# In[40]:


data=pd.DataFrame(d)
# data


# In[41]:


plt.figure(figsize=(10, 6))
plt.hist(data['Dur_msec'] / 1000, bins=30, edgecolor='black')
plt.xlabel('Session Duration (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Session Duration')
plt.show()


# # Session duration

# In[42]:


telecom_data = data.groupby('Xdr_Session').agg({
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()
print("duration",telecom_data)


# # total download (DL) and upload (UL) data

# In[43]:


total_data =df["Total UL (Bytes)"] + df["Total DL (Bytes)"]
print(total_data)


# # total data volume (in Bytes) during this session for each application

# In[44]:


total_data_rows = df[['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
       'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
       'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
       'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
       'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']].sum(axis=0)
print("total_data_rows : ")

print(total_data_rows)


# # Task 2  - User Engagement Analysis

# # The duration of the sessions

# In[45]:


total_duration = df["Dur. (ms)"] + df["Dur. (ms).1"] + df["Activity Duration DL (ms)"] + df["Activity Duration UL (ms)"] 

print(total_duration)


# # the session total traffic (download and upload (bytes)

# In[46]:


total_sessions  = df[['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
       'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
       'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
       'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
       'Gaming UL (Bytes)',
        'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
        'HTTP DL (Bytes)', 'HTTP UL (Bytes)',
        'Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=0)
print("total sessions :", total_sessions)


# In[ ]:


# Plotting the total data volume for each application


# In[47]:


applications = total_sessions.index
data_volume = total_sessions.values

plt.figure(figsize=(10, 6))
plt.bar(applications, data_volume,)
plt.xlabel('Application')
plt.ylabel('Data Volume (Bytes)')
plt.title('Total Data Volume for Each Application')
plt.xticks(rotation=45)

plt.show()


# # Task 1.2

# In[48]:


# Conduct exploratory data analysis on those data & communicate useful insights.
# Ensure that you identify and treat all missing values and outliers in the dataset by replacing them with the mean of the corresponding column.


# In[51]:


for i in data.columns:
    
    if data[i].dtype == 'object':
        mode_value = data[i].mode()[0] 
        data[i].fillna(mode_value, inplace=True)
    else:
        mean_value = data[i].mean()
        data[i].fillna(mean_value, inplace=True) 


# In[50]:


data.isna().sum()


# In[54]:


data_filled = data.fillna(0)
# data_filled


# In[55]:


plt.hist(data_filled['Dur_msec'], bins=20)
plt.xlabel('Duration (ms)')
plt.ylabel('Count')
plt.title('Distribution of Dur_msec')
plt.show()


# # Univariate Analysis--- dispersion parameters for each quantitative variable:
# 

# In[ ]:


# Analyze the basic metrics (mean, median, etc)


# In[57]:


quantitative_vars = ['Dur_msec', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)', 'Social Media DL (Bytes)', 
                     'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)', 
                     'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 
                     'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 
                     'Other UL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)']


# In[58]:


dispersion_stats = pd.DataFrame(columns=['Variable', 'Mean', 'Median', 'Range', 'Interquartile Range', 'Standard Deviation'])


# In[59]:


for var in quantitative_vars:
    stats = {}
    stats['Variable'] = var
    stats['Mean'] = data[var].mean()
    stats['Median'] = data[var].median()
    stats['Range'] = data[var].max() - data[var].min()
    stats['Interquartile Range'] = data[var].quantile(0.75) - data[var].quantile(0.25)
    stats['Standard Deviation'] = data[var].std()

    dispersion_stats = dispersion_stats.append(stats, ignore_index=True)

#The dispersion parameters for each variable:
print("Dispersion Parameters:")
print(dispersion_stats)


# # Plot histograms for each quantitative variable
# 

# In[64]:


plt.figure(figsize=(16, 8))
for i, var in enumerate(quantitative_vars):
    plt.subplot(5, 4, i+1)
    plt.hist(data[var], bins=30, edgecolor='black')
    plt.xlabel(var)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# # Bivariate Analysis---- correlation matrix as a heatmap
# 

# In[65]:


variables = ['Dur_msec', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)',
             'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
             'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
             'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 
             'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
             'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)', 
             'Total UL (Bytes)', 'Total DL (Bytes)']


# In[67]:


correlation_matrix = data[variables].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[68]:


non_categorical_columns=[]
for col in data.columns:
    if data[col].dtypes != "object":
        non_categorical_columns.append(col)


# In[69]:


non_categorical_columns


# In[70]:


data.skew()


# In[71]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[72]:


features = ['Dur_msec', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)', 'Social Media DL (Bytes)',
            'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
            'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)',
            'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)',
            'Other UL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)']


# In[73]:


# Calculate the skewness of each feature


# In[74]:


skewness = data.skew()


# In[75]:


# Plot the density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(skewness, shade=True)
plt.xlabel('Skewness')
plt.ylabel('Density')
plt.title('Density Plot of Skewness')
plt.show()


# In[ ]:




