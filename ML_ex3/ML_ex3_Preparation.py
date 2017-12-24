
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# ## Load Data:

# In[2]:


df = pd.read_csv("ElectionsData.csv", header=0)


# ## Drop Features:

# In[3]:


col_list = ['Number_of_valued_Kneset_members', 'Yearly_IncomeK', 'Overall_happiness_score',
'Avg_Satisfaction_with_previous_vote', 'Most_Important_Issue', 'Will_vote_only_large_party', 
'Garden_sqr_meter_per_person_in_residancy_area', 'Weighted_education_rank']
df = df[col_list]


# In[4]:


print('Number of features: '+str(len(df.columns.values)))
df.dtypes


# ## Data Preperation:

# #### 1. Number_of_valued_Kneset_members

# In[5]:


print('Null values: '+str(df['Number_of_valued_Kneset_members'].isnull().sum()))
df['Number_of_valued_Kneset_members'].hist(bins=1)
df['Number_of_valued_Kneset_members'].describe()


# In[6]:


df['Number_of_valued_Kneset_members_isNull'] = pd.isnull(df['Number_of_valued_Kneset_members']).astype(int)
df['Number_of_valued_Kneset_members'] = df['Number_of_valued_Kneset_members'].interpolate()
print('Null values after imputation: '+str(df['Number_of_valued_Kneset_members'].isnull().sum()))


# #### 2. Yearly_IncomeK

# In[7]:


print('Null values: '+str(df['Yearly_IncomeK'].isnull().sum()))
df['Yearly_IncomeK'].hist()
df['Yearly_IncomeK'].describe()


# In[8]:


df['Yearly_IncomeK_isNull'] = pd.isnull(df.Yearly_IncomeK).astype(int)
avg = df['Yearly_IncomeK'].mean()
df['Yearly_IncomeK'] = df['Yearly_IncomeK'].fillna(value = avg)
print('Null values after imputation: '+str(df['Yearly_IncomeK'].isnull().sum()))


# #### 3. Overall_happiness_score

# In[9]:


print('Null values: '+str(df['Overall_happiness_score'].isnull().sum()))
df['Overall_happiness_score'].hist(bins=50)
df['Overall_happiness_score'].describe()


# In[10]:


df['Overall_happiness_score_isNull'] = pd.isnull(df['Overall_happiness_score']).astype(int)
df['Overall_happiness_score'] = df['Overall_happiness_score'].interpolate()
print('Null values after imputation: '+str(df['Overall_happiness_score'].isnull().sum()))


# #### 4. Avg_Satisfaction_with_previous_vote

# In[11]:


print('Null values: '+str(df['Avg_Satisfaction_with_previous_vote'].isnull().sum()))
df['Avg_Satisfaction_with_previous_vote'].hist(bins=40)
df['Avg_Satisfaction_with_previous_vote'].describe()


# In[12]:


df['Avg_Satisfaction_with_previous_vote_isNull'] = pd.isnull(df.Avg_Satisfaction_with_previous_vote).astype(int)
med = df['Avg_Satisfaction_with_previous_vote'].median()
df['Avg_Satisfaction_with_previous_vote'] = df['Avg_Satisfaction_with_previous_vote'].fillna(value = med)
print('Null values after imputation: '+str(df['Avg_Satisfaction_with_previous_vote'].isnull().sum()))


# #### 5. Garden_sqr_meter_per_person_in_residancy_area

# In[13]:


print('Null values: '+str(df['Garden_sqr_meter_per_person_in_residancy_area'].isnull().sum()))
df['Garden_sqr_meter_per_person_in_residancy_area'].hist()
df['Garden_sqr_meter_per_person_in_residancy_area'].describe()


# In[14]:


df['Garden_sqr_meter_per_person_in_residancy_area_isNull'] = pd.isnull(df.Garden_sqr_meter_per_person_in_residancy_area).astype(int)
avg = df['Garden_sqr_meter_per_person_in_residancy_area'].mean()
df['Garden_sqr_meter_per_person_in_residancy_area'] = df['Garden_sqr_meter_per_person_in_residancy_area'].fillna(value = avg)
print('Null values after imputation: '+str(df['Garden_sqr_meter_per_person_in_residancy_area'].isnull().sum()))


# #### 6. Weighted_education_rank

# In[15]:


print('Null values: '+str(df['Weighted_education_rank'].isnull().sum()))
df['Weighted_education_rank'].hist()
df['Weighted_education_rank'].describe()


# In[16]:


df['Weighted_education_rank_isNull'] = pd.isnull(df.Garden_sqr_meter_per_person_in_residancy_area).astype(int)
med = df['Weighted_education_rank'].median()
df['Weighted_education_rank'] = df['Weighted_education_rank'].fillna(value = med)
print('Null values after imputation: '+str(df['Weighted_education_rank'].isnull().sum()))


# #### 7. Most_Important_Issue

# In[17]:


df['Most_Important_Issue'].value_counts().plot(kind='bar')
print('Null values: '+str(df['Most_Important_Issue'].isnull().sum()))
df['Most_Important_Issue_cat'] = df['Most_Important_Issue'].astype("category")
df['Most_Important_Issue_int'] = df['Most_Important_Issue_cat'].cat.rename_categories(range(df['Most_Important_Issue_cat'].nunique())).astype(int)
print('Unique values: '+str(df['Most_Important_Issue_int'].unique()))


# In[18]:


print('Null values: '+str(df['Most_Important_Issue_int'].isnull().sum()))
df['Most_Important_Issue_int'].replace(to_replace=-9223372036854775808, value=8, inplace=True)
df['Most_Important_Issue_int'].hist()
df['Most_Important_Issue_int'].unique()
neigh = KNeighborsClassifier(n_neighbors=1)
df_num = df.loc[df['Most_Important_Issue_int']<8].dropna().select_dtypes(include=['float64'])
X = df_num
y = df.loc[df['Most_Important_Issue_int']<8].dropna().Most_Important_Issue_int
neigh.fit(X, y) 
for index, row in df.dropna().loc[df['Most_Important_Issue_int']>7].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
#     print(neighbor)
    df.Most_Important_Issue_int[index] = neighbor
df['Most_Important_Issue_int'].replace(to_replace=8, value=7, inplace=True)
df['Most_Important_Issue_int'].unique()


# #### 8. Will_vote_only_large_party

# In[19]:


df['Will_vote_only_large_party'].value_counts().plot(kind='bar')
print('Null values: '+str(df['Will_vote_only_large_party'].isnull().sum()))
df['Will_vote_only_large_party_cat'] = df['Will_vote_only_large_party'].astype("category")
df['Will_vote_only_large_party_int'] = df['Will_vote_only_large_party_cat'].cat.rename_categories(range(df['Will_vote_only_large_party_cat'].nunique())).astype(int)
print('Unique values: '+str(df['Will_vote_only_large_party_int'].unique()))


# In[20]:


print('Null values: '+str(df['Will_vote_only_large_party_int'].isnull().sum()))
df['Will_vote_only_large_party_int'].replace(to_replace=-9223372036854775808, value=3, inplace=True)
df['Will_vote_only_large_party_int'].hist()
df['Will_vote_only_large_party_int'].unique()
neigh = KNeighborsClassifier(n_neighbors=1)
df_num = df.loc[df['Will_vote_only_large_party_int']<3].dropna().select_dtypes(include=['float64'])
X = df_num
y = df.loc[df['Will_vote_only_large_party_int']<3].dropna().Will_vote_only_large_party_int
neigh.fit(X, y) 
for index, row in df.dropna().loc[df['Will_vote_only_large_party_int']>2].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
#     print(neighbor)
    df.Will_vote_only_large_party_int[index] = neighbor
df['Will_vote_only_large_party_int'].replace(to_replace=3, value=1, inplace=True)
df['Will_vote_only_large_party_int'].unique()


# In[21]:


df.dtypes


# In[22]:


df.isnull().sum().sort_values(ascending=False)


# In[23]:


df = df.drop(['Most_Important_Issue','Most_Important_Issue_cat','Will_vote_only_large_party','Will_vote_only_large_party_cat'],1)


# In[24]:


df.isnull().sum().sort_values(ascending=False)


# #### Apply scaling:

# In[25]:


N = ['Yearly_IncomeK','Overall_happiness_score','Avg_Satisfaction_with_previous_vote','Garden_sqr_meter_per_person_in_residancy_area']
normalScaler = StandardScaler()
df[N] = normalScaler.fit_transform(df[N])
U = ['Number_of_valued_Kneset_members']
minMaxScaler = MinMaxScaler()
df[U] = minMaxScaler.fit_transform(df[U])


# ##  Split data to train, test and validation:

# In[26]:


train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])


# ## Save to csv:

# In[27]:


train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
validate.to_csv("validate.csv", index=False)

