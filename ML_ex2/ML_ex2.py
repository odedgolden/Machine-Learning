
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE



data = pd.read_csv("ElectionsData.csv", header=0)

data['Looking_at_poles_results_int'] = data['Looking_at_poles_results'].map( {'No':0, 'Yes':1, np.nan:-1}).astype(int)# In[99]:

data['Married_int'] = data['Married'].map( {'No':0, 'Yes':1, np.nan:-1}).astype(int)

data['Gender_int'] = data['Gender'].map( {'Male':0, 'Female':1, np.nan:-1}).astype(int)

data['Voting_Time_int'] = data['Voting_Time'].map( {'By_16:00':0, 'After_16:00':1, np.nan:-1}).astype(int)

data['Financial_agenda_matters_int'] = data['Financial_agenda_matters'].map( {'No':0, 'Yes':1, np.nan:-1}).astype(int)

data['Most_Important_Issue_cat'] = data['Most_Important_Issue'].astype("category")
data['Most_Important_Issue_int'] = data['Most_Important_Issue_cat'].cat.rename_categories(range(data['Most_Important_Issue_cat'].nunique())).astype(int)

data['Age_group_cat'] = data['Age_group'].astype("category")
data['Age_group_int'] = data['Age_group_cat'].cat.rename_categories(range(data['Age_group_cat'].nunique())).astype(int)

data['Will_vote_only_large_party_cat'] = data['Will_vote_only_large_party'].astype("category")
data['Will_vote_only_large_party_int'] = data['Will_vote_only_large_party_cat'].cat.rename_categories(range(data['Will_vote_only_large_party_cat'].nunique())).astype(int)

data['Main_transportation_cat'] = data['Main_transportation'].astype("category")
data['Main_transportation_int'] = data['Main_transportation_cat'].cat.rename_categories(range(data['Main_transportation_cat'].nunique())).astype(int)

data['Occupation_cat'] = data['Occupation'].astype("category")
data['Occupation_int'] = data['Occupation_cat'].cat.rename_categories(range(data['Occupation_cat'].nunique())).astype(int)

data = data.drop(['Looking_at_poles_results','Married','Gender','Voting_Time','Financial_agenda_matters','Most_Important_Issue','Most_Important_Issue_cat','Age_group','Age_group_cat','Will_vote_only_large_party','Will_vote_only_large_party_cat','Main_transportation','Will_vote_only_large_party_cat','Main_transportation','Main_transportation_cat','Occupation','Occupation_cat'], axis=1) 

data['Avg_environmental_importance_isNull'] = pd.isnull(data.Avg_environmental_importance).astype(int)
data['Avg_environmental_importance'] = data['Avg_environmental_importance'].interpolate()

data['Avg_education_importance_isNull'] = pd.isnull(data.Avg_environmental_importance).astype(int)
data['Avg_education_importance'] = data['Avg_education_importance'].interpolate()

data['Yearly_ExpensesK_isNull'] = pd.isnull(data.Yearly_ExpensesK).astype(int)
data['Yearly_ExpensesK'] = data['Yearly_ExpensesK'].interpolate()

data['Garden_sqr_meter_per_person_in_residancy_area_isNull'] = pd.isnull(data.Garden_sqr_meter_per_person_in_residancy_area).astype(int)
avg = data['Garden_sqr_meter_per_person_in_residancy_area'].mean()
data['Garden_sqr_meter_per_person_in_residancy_area'] = data['Garden_sqr_meter_per_person_in_residancy_area'].fillna(value = avg)

data['Num_of_kids_born_last_10_years_isNull'] = pd.isnull(data.Garden_sqr_meter_per_person_in_residancy_area).astype(int)
med = data['Num_of_kids_born_last_10_years'].median()
data['Num_of_kids_born_last_10_years'] = data['Num_of_kids_born_last_10_years'].fillna(value = med)

data['%Time_invested_in_work_isNull'] = pd.isnull(data.Yearly_ExpensesK).astype(int)
data['%Time_invested_in_work'] = data['%Time_invested_in_work'].interpolate()

data['Weighted_education_rank_isNull'] = pd.isnull(data.Garden_sqr_meter_per_person_in_residancy_area).astype(int)
med = data['Weighted_education_rank'].median()
data['Weighted_education_rank'] = data['Weighted_education_rank'].fillna(value = med)

data['Avg_monthly_expense_on_pets_or_plants_isNull'] = pd.isnull(data.Avg_monthly_expense_on_pets_or_plants).astype(int)
avg = data['Avg_monthly_expense_on_pets_or_plants'].mean()
data['Avg_monthly_expense_on_pets_or_plants'] = data['Avg_monthly_expense_on_pets_or_plants'].fillna(value = avg)

data['%_satisfaction_financial_policy_isNull'] = pd.isnull(data['%_satisfaction_financial_policy']).astype(int)
data['%_satisfaction_financial_policy'] = data['%_satisfaction_financial_policy'].interpolate()

data['Political_interest_Total_Score_isNull'] = pd.isnull(data.Political_interest_Total_Score).astype(int)
med = data['Political_interest_Total_Score'].median()
data['Political_interest_Total_Score'] = data['Political_interest_Total_Score'].fillna(value = med)

data['Avg_size_per_room_isNull'] = pd.isnull(data.Avg_size_per_room).astype(int)
med = data['Avg_size_per_room'].median()
data['Avg_size_per_room'] = data['Avg_size_per_room'].fillna(value = med)

data['Avg_government_satisfaction_isNull'] = pd.isnull(data.Avg_government_satisfaction).astype(int)
data['Avg_government_satisfaction'] = data['Avg_government_satisfaction'].interpolate()

data['Occupation_Satisfaction_isNull'] = pd.isnull(data.Occupation_Satisfaction).astype(int)
data['Occupation_Satisfaction'] = data['Occupation_Satisfaction'].interpolate()

data['Avg_monthly_expense_when_under_age_21_isNull'] = pd.isnull(data.Avg_monthly_expense_when_under_age_21).astype(int)
med = data['Avg_monthly_expense_when_under_age_21'].median()
data['Avg_monthly_expense_when_under_age_21'] = data['Avg_monthly_expense_when_under_age_21'].fillna(value = med)

data['Financial_balance_score_(0-1)_isNull'] = pd.isnull(data['Financial_balance_score_(0-1)']).astype(int)
data['Financial_balance_score_(0-1)'] = data['Financial_balance_score_(0-1)'].interpolate()

data['Avg_monthly_household_cost_isNull'] = pd.isnull(data.Avg_monthly_household_cost).astype(int)
avg = data['Avg_monthly_household_cost'].mean()
data['Avg_monthly_household_cost'] = data['Avg_monthly_household_cost'].fillna(value = avg)

data['Yearly_IncomeK_isNull'] = pd.isnull(data.Yearly_IncomeK).astype(int)
avg = data['Yearly_IncomeK'].mean()
data['Yearly_IncomeK'] = data['Yearly_IncomeK'].fillna(value = avg)

data['%Of_Household_Income_isNull'] = pd.isnull(data['%Of_Household_Income']).astype(int)
data['%Of_Household_Income'] = data['%Of_Household_Income'].interpolate()

data['Number_of_valued_Kneset_members_isNull'] = pd.isnull(data['Number_of_valued_Kneset_members']).astype(int)
data['Number_of_valued_Kneset_members'] = data['Number_of_valued_Kneset_members'].interpolate()

data['AVG_lottary_expanses_isNull'] = pd.isnull(data.AVG_lottary_expanses).astype(int)
med = data['AVG_lottary_expanses'].median()
data['AVG_lottary_expanses'] = data['AVG_lottary_expanses'].fillna(value = med)

data['Overall_happiness_score_isNull'] = pd.isnull(data['Overall_happiness_score']).astype(int)
data['Overall_happiness_score'] = data['Overall_happiness_score'].interpolate()

data['Phone_minutes_10_years_isNull'] = pd.isnull(data['Phone_minutes_10_years']).astype(int)
data['Phone_minutes_10_years'] = data['Phone_minutes_10_years'].interpolate()

data['Avg_Residancy_Altitude_isNull'] = pd.isnull(data['Avg_Residancy_Altitude']).astype(int)
data['Avg_Residancy_Altitude'] = data['Avg_Residancy_Altitude'].interpolate()

data['Avg_monthly_income_all_years_isNull'] = pd.isnull(data.Avg_monthly_income_all_years).astype(int)
med = data['Avg_monthly_income_all_years'].median()
data['Avg_monthly_income_all_years'] = data['Avg_monthly_income_all_years'].fillna(value = med)

data['Avg_Satisfaction_with_previous_vote_isNull'] = pd.isnull(data.Avg_Satisfaction_with_previous_vote).astype(int)
med = data['Avg_Satisfaction_with_previous_vote'].median()
data['Avg_Satisfaction_with_previous_vote'] = data['Avg_Satisfaction_with_previous_vote'].fillna(value = med)

data['Last_school_grades_isNull'] = pd.isnull(data.Last_school_grades).astype(int)
med = data['Last_school_grades'].median()
data['Last_school_grades'] = data['Last_school_grades'].fillna(value = med)

data['Number_of_differnt_parties_voted_for_isNull'] = pd.isnull(data.Number_of_differnt_parties_voted_for).astype(int)
med = data['Number_of_differnt_parties_voted_for'].median()
data['Number_of_differnt_parties_voted_for'] = data['Number_of_differnt_parties_voted_for'].fillna(value = med)

data['Occupation_int'].replace(to_replace=-9223372036854775808, value=5, inplace=True)

neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Occupation_int']<5].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Occupation_int']<5].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Occupation_int']>4].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Occupation_int'].replace(to_replace=5, value=2, inplace=True)
data['Occupation_int'].unique()

data['Main_transportation_int'].replace(to_replace=-9223372036854775808, value=4, inplace=True)
neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Main_transportation_int']<4].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Main_transportation_int']<4].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Main_transportation_int']>3].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Main_transportation_int'].replace(to_replace=4, value=2, inplace=True)
data['Main_transportation_int'].unique()

neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Looking_at_poles_results_int']>-1].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Looking_at_poles_results_int']>-1].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Looking_at_poles_results_int']<0].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Looking_at_poles_results_int'].replace(to_replace=-1, value=0, inplace=True)
data['Looking_at_poles_results_int'].unique()

neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Married_int']>-1].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Married_int']>-1].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Married_int']<0].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Married_int'].replace(to_replace=-1, value=0, inplace=True)
data['Married_int'].unique()

neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Gender_int']>-1].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Gender_int']>-1].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Gender_int']<0].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Gender_int'].replace(to_replace=-1, value=0, inplace=True)
data['Gender_int'].unique()

neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Voting_Time_int']>-1].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Voting_Time_int']>-1].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Voting_Time_int']<0].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Voting_Time_int'].replace(to_replace=-1, value=0, inplace=True)
data['Voting_Time_int'].unique()

neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Financial_agenda_matters_int']>-1].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Financial_agenda_matters_int']>-1].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Financial_agenda_matters_int']<0].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Financial_agenda_matters_int'].replace(to_replace=-1, value=0, inplace=True)
data['Financial_agenda_matters_int'].unique()

data['Most_Important_Issue_int'].replace(to_replace=-9223372036854775808, value=8, inplace=True)

neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Most_Important_Issue_int']<8].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Most_Important_Issue_int']<8].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Most_Important_Issue_int']>7].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Most_Important_Issue_int'].replace(to_replace=8, value=7, inplace=True)
data['Most_Important_Issue_int'].unique()

data['Age_group_int'].replace(to_replace=-9223372036854775808, value=3, inplace=True)
neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Age_group_int']<3].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Age_group_int']<3].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Age_group_int']>2].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Age_group_int'].replace(to_replace=3, value=1, inplace=True)
data['Age_group_int'].unique()

data['Will_vote_only_large_party_int'].replace(to_replace=-9223372036854775808, value=3, inplace=True)
neigh = KNeighborsClassifier(n_neighbors=1)
df_num = data.loc[data['Will_vote_only_large_party_int']<3].dropna().select_dtypes(include=['float64'])
X = df_num
y = data.loc[data['Will_vote_only_large_party_int']<3].dropna().Occupation_int
neigh.fit(X, y) 
for index, row in data.dropna().loc[data['Will_vote_only_large_party_int']>2].iterrows():
    row_with_only_floats = [[x for x in row if np.dtype(type(x))==np.float64]]
    neighbor = neigh.predict(np.array(row_with_only_floats))[0]
    data.Occupation_int[index] = neighbor
data['Will_vote_only_large_party_int'].replace(to_replace=3, value=1, inplace=True)
data['Will_vote_only_large_party_int'].unique()


N = ['Avg_monthly_expense_when_under_age_21','AVG_lottary_expanses','Garden_sqr_meter_per_person_in_residancy_area','Yearly_IncomeK','Avg_monthly_expense_on_pets_or_plants','Avg_monthly_household_cost','Will_vote_only_large_party_int','Avg_size_per_room','Number_of_differnt_parties_voted_for','Political_interest_Total_Score','Overall_happiness_score']
normalScaler = StandardScaler()
data[N] = normalScaler.fit_transform(data[N])

U = ['Occupation_Satisfaction','Most_Important_Issue_int','Avg_Satisfaction_with_previous_vote','Financial_balance_score_(0-1)','%Of_Household_Income','Avg_government_satisfaction','Avg_education_importance','Avg_environmental_importance','Avg_Residancy_Altitude','Yearly_ExpensesK','%Time_invested_in_work','Phone_minutes_10_years','Weighted_education_rank','%_satisfaction_financial_policy','Avg_monthly_income_all_years','Last_school_grades','Age_group_int','Number_of_valued_Kneset_members','Main_transportation_int','Occupation_int','Num_of_kids_born_last_10_years']
minMaxScaler = MinMaxScaler()
data[U] = minMaxScaler.fit_transform(data[U])

y = data['Vote']
X = data.drop(['Vote'],1)

sel = VarianceThreshold()
X = sel.fit_transform(X,y)
print(X.shape)

model = ExtraTreesClassifier()
model.fit(X,y)
# display the relative importance of each attribute
print("Features result after Extra Trees Classifier: \n"+str(model.feature_importances_)+"\n\n")


model = LogisticRegression()
# create the RFE model and select 20 attributes
rfe = RFE(model, 20)
rfe = rfe.fit(X,y)
print("Features result after RFE: \n"+str(rfe.support_)+"\n\n")
print(rfe.ranking_)


sel = SelectKBest(f_classif, k=20)
res = sel.fit_transform(X,y)
print("Results of select k best: \n"+str(res.unique()))


train, validate, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
validate.to_csv("validate.csv", index=False)

