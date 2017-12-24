
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd


# ## Load Data:

# In[13]:


df = pd.read_csv("ElectionsData.csv", header=0)


# ## Drop Features:

# In[16]:


col_list = ['Number_of_valued_Kneset_members', 'Yearly_IncomeK', 'Overall_happiness_score',
'Avg_Satisfaction_with_previous_vote', 'Most_Important_Issue', 'Will_vote_only_large_party', 
'Garden_sqr_meter_per_person_in_residancy_area', 'Weighted_education_rank']
df = df[col_list]


# In[18]:


print('Number of features: '+str(len(df.columns.values)))
df.dtypes


# ## Data Preperation:

# In[19]:


print('Null values: '+str(data['Number_of_valued_Kneset_members'].isnull().sum()))
data['Number_of_valued_Kneset_members'].hist(bins=1)
data['Number_of_valued_Kneset_members'].describe()


# In[20]:


print('Null values: '+str(data['Yearly_IncomeK'].isnull().sum()))
data['Yearly_IncomeK'].hist()
data['Yearly_IncomeK'].describe()


# In[21]:


print('Null values: '+str(data['Overall_happiness_score'].isnull().sum()))
data['Overall_happiness_score'].hist(bins=50)
data['Overall_happiness_score'].describe()


# In[22]:


print('Null values: '+str(data['Avg_Satisfaction_with_previous_vote'].isnull().sum()))
data['Avg_Satisfaction_with_previous_vote'].hist(bins=40)
data['Avg_Satisfaction_with_previous_vote'].describe()


# In[24]:


data['Most_Important_Issue'].value_counts().plot(kind='bar')
print('Null values: '+str(data['Most_Important_Issue'].isnull().sum()))
data['Most_Important_Issue_cat'] = data['Most_Important_Issue'].astype("category")
data['Most_Important_Issue_int'] = data['Most_Important_Issue_cat'].cat.rename_categories(range(data['Most_Important_Issue_cat'].nunique())).astype(int)
print('Unique values: '+str(data['Most_Important_Issue_int'].unique()))


# In[25]:


print('Null values: '+str(data['Most_Important_Issue_int'].isnull().sum()))
data['Most_Important_Issue_int'].replace(to_replace=-9223372036854775808, value=8, inplace=True)
data['Most_Important_Issue_int'].hist()
data['Most_Important_Issue_int'].unique()


# In[27]:


data['Will_vote_only_large_party'].value_counts().plot(kind='bar')
print('Null values: '+str(data['Will_vote_only_large_party'].isnull().sum()))
data['Will_vote_only_large_party_cat'] = data['Will_vote_only_large_party'].astype("category")
data['Will_vote_only_large_party_int'] = data['Will_vote_only_large_party_cat'].cat.rename_categories(range(data['Will_vote_only_large_party_cat'].nunique())).astype(int)
print('Unique values: '+str(data['Will_vote_only_large_party_int'].unique()))


# In[28]:


print('Null values: '+str(data['Will_vote_only_large_party_int'].isnull().sum()))
data['Will_vote_only_large_party_int'].replace(to_replace=-9223372036854775808, value=3, inplace=True)
data['Will_vote_only_large_party_int'].hist()
data['Will_vote_only_large_party_int'].unique()


# In[29]:


print('Null values: '+str(data['Garden_sqr_meter_per_person_in_residancy_area'].isnull().sum()))
data['Garden_sqr_meter_per_person_in_residancy_area'].hist()
data['Garden_sqr_meter_per_person_in_residancy_area'].describe()


# In[30]:


print('Null values: '+str(data['Weighted_education_rank'].isnull().sum()))
data['Weighted_education_rank'].hist()
data['Weighted_education_rank'].describe()

