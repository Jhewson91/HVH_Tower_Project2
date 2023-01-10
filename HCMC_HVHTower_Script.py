# IMPORT LIBRARIES
import warnings
import pandas as pd                 # For data analysis
import matplotlib.pyplot as plt     # For basic visualization
import seaborn as sns               # For advanced visualization
import numpy as np                  # For mathematical operations
import missingno as msno            # For missing values visualizaiton
from sklearn import linear_model    # For linear regression model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# Import Dataset
df = pd.read_csv(
    r'C:\Users\jhews\OneDrive\Documents\Data Analytics\Portfolio\Project 1 - HCMC Office Sector Database\Office Sector Market Info 2020.csv')

# View column titles
print(df.columns)

# View first 10 rows
print(df.head(10))
df.drop(["Unnamed"], axis=1, inplace=True)
print(df.head(10))

# DATA PREPREPARATION

# (1) Identify missing values

# Discover missing values
# Identify if there are any missing values in each variable
print(df.isnull().any())
print(df.isnull().sum())  # Count the sum of missing values in each variable
print('Row count is:', len(df.index))  # Count how many rows there are in total
# Calculate the % of rows missing in each variable to determine suitable data cleaning method
for column in df.columns:
    percentage = df[column].isnull().mean()
    print(f'{column}: {round(percentage*100, 2)}%')

# Visualize missing data with heatmap
# msno.matrix(df)
#plt.title('Matrix showing missing values in Transaction Dataset')
# plt.show()

# Drop rows with missing values (only small percentage missing so delete missing values)
newdf = df.dropna()
# Check this has worked
print(newdf.isnull().any())
# msno.matrix(newdf)
#plt.title('Matrix showing missing values in Transaction Dataset')
# plt.show()

# (2) Identify Inconistent values:
# get a list of unique strings in'transaction_building' column
trans_bld = df['transaction_building'].unique()
print(trans_bld)
# sort them alphabetically and then take a closer look
trans_bld.sort()
print(trans_bld)
print(newdf['transaction_building'].value_counts())

# Drop inconsistent row values
newdf = newdf[newdf['transaction_building'].str.contains(
    "Empire City 1 ") == False]
newdf = newdf[newdf['transaction_building'].str.contains(
    "Saigon Centre 20") == False]
newdf = newdf[newdf['transaction_building'].str.contains(
    "Thai holdings Tower") == False]
newdf = newdf[newdf['transaction_building'].str.contains(
    "TechHub Saigond") == False]
print(newdf['transaction_building'].value_counts())

# (3) Identify Duplications
duplicate = newdf[newdf.duplicated()]
print("Duplicate Rows :")
print(duplicate)
# Drop duplications from dataset
drop_duplicate = newdf.drop_duplicates(subset="tenant_id",
                                       keep=False, inplace=True)
print(drop_duplicate)

# (4) Identify Outliers:
# (i) "Net Effective Transactiing Rental Rates" Boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=newdf['net_effective_rr_$/sqm']).set(
    title='Boxplot showing Net Effective Transacting Rental Rates results')
plt.show()

# Identified "Net Effective Transacting Rental Rates" outliers from Box Plot and drop outliers
newdf = newdf[(newdf['net_effective_rr_$/sqm'] < 76.0)]

# (ii) market_vacancy_% Stats Summary:
print(newdf['market_vacancy_%'].describe())
# "Market Vacancy(%)" Histogram
newf = newdf.astype({'market_vacancy_%': 'int'})
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(newdf['market_vacancy_%'])
plt.xlabel("Market Vacancy(%)")
plt.ylabel("Frequency")
plt.title("Frequency of Market Vacancy(%)")
plt.show()

# Identified "market_vacancy_%" outliers from Histogram and drop outliers
newdf = newdf[(newdf['market_vacancy_%'] > 1.0)]


# (iii) sub_sector_vacancy_% Stats Summary:
print(newdf['sub_sector_vacancy_%'].describe())
# "Market Vacancy(%)" Histogram
newf = newdf.astype({'sub_sector_vacancy_%': 'int'})
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(newdf['sub_sector_vacancy_%'])
plt.xlabel("Sub-sector Vacancy(%)")
plt.ylabel("Frequency")
plt.title("Frequency of Sub-sector Vacancy(%)")
plt.show()

# Identified "sub_sector_vacancy_%" outliers from Histogram and drop outliers
newdf = newdf[(newdf['sub_sector_vacancy_%'] > 1.0)]


# PRELIMINARY VISUALIZATION - Numeric Variables

# Net Effective Transactiing Rental Rates

# Net Effective Transactiing Rental Rates Stats Summary:
print(newdf['net_effective_rr_$/sqm'].describe())
# "Net Effective Transactiing Rental Rates" Histogram
newf = newdf.astype({'net_effective_rr_$/sqm': 'int'})
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(newdf['net_effective_rr_$/sqm'])
plt.xlabel("Net Effective Transacting Rental Rate (US$/sqm/annum")
plt.ylabel("Frequency")
plt.title("Frequency of Net Effective Transacting Rental Rates")
plt.show()

# "Net Effective Transactiing Rental Rates" Boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=newdf['net_effective_rr_$/sqm']).set(
    title='Boxplot showing Net Effective Transacting Rental Rates results')
plt.show()


# Market Vacancy(%)

# market_vacancy_% Stats Summary:
print(newdf['market_vacancy_%'].describe())
# "Market Vacancy(%)" Histogram
newf = newdf.astype({'market_vacancy_%': 'int'})
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(newdf['market_vacancy_%'])
plt.xlabel("Market Vacancy(%)")
plt.ylabel("Frequency")
plt.title("Frequency of Market Vacancy(%)")
plt.show()

# "market_vacancy_%" Boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=newdf['market_vacancy_%']).set(
    title='Boxplot showing Market Vacancy(%)')
plt.show()


# Sub-sector Vacancy(%)

# sub_sector_vacancy_% Stats Summary:
print(newdf['sub_sector_vacancy_%'].describe())
# "Market Vacancy(%)" Histogram
newf = newdf.astype({'sub_sector_vacancy_%': 'int'})
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.hist(newdf['sub_sector_vacancy_%'])
plt.xlabel("Sub-sector Vacancy(%)")
plt.ylabel("Frequency")
plt.title("Frequency of Sub-sector Vacancy(%)")
plt.show()

# "market_vacancy_%" Boxplot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=newdf['sub_sector_vacancy_%']).set(
    title='Boxplot showing Sub-sector Vacancy(%)')
plt.show()

# "Market Supply (million sqm)" Boxplot
print(newdf['market_supply_sqm'].describe())
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.boxplot(x=newdf['market_supply_sqm']).set(
    title='Boxplot showing Market Supply (million sqm)')
plt.show()


# PRELIMINARY VISUALIZATION - Categorical Variables
newdf.columns = newdf.columns.str.strip()

# Bar chart showing 'transaction_date'
fig = newdf['transaction_date'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of transaction_date')
plt.xlabel('transaction_date')
plt.ylabel('Frequency of transaction_date')
plt.show()

# Bar chart showing 'tenant_industry'
fig = newdf['tenant_industry'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of tenant_industry')
plt.xlabel('tenant_industry')
plt.ylabel('Frequency of tenant_industry')
plt.show()

# Bar chart showing 'transaction_building'
fig = newdf['transaction_building'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of transaction_building')
plt.xlabel('transaction_building')
plt.ylabel('Frequency of transaction_building')
plt.show()

# Bar chart showing 'building_location'
fig = newdf['building_location'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of building_location')
plt.xlabel('building_location')
plt.ylabel('Frequency of building_location')
plt.show()

# Bar chart showing 'building_class'
fig = newdf['building_class'].value_counts().plot.bar().get_figure()
plt.title('Bar Chart showing Frequency of building_class')
plt.xlabel('building_class')
plt.ylabel('Frequency of building_class')
plt.show()


# Data Preparation
newdf.columns = newdf.columns.str.strip()
# Drop non-relevant variables from data set
newdf.drop(['tenant_id', 'transaction_type', 'lease_length_years', 'tenant_industry', 'transaction_date',
           'building_location', 'transaction_building', 'market_vacancy_sqm', 'market_supply_sqm'], axis=1, inplace=True)
print(newdf.head(10))

# get a list of unique strings in''building_class' column
bld_class = newdf['building_class'].unique()
print(bld_class)

# Remove non-relevant building classifications
newdf = newdf[newdf['building_class'].str.contains("Grade B") == False]
newdf = newdf[newdf['building_class'].str.contains("District 2") == False]
newdf = newdf[newdf['building_class'].str.contains("District 7") == False]
newdf = newdf[newdf['building_class'].str.contains("District 3") == False]
newdf = newdf[newdf['building_class'].str.contains("District 4") == False]
newdf = newdf[newdf['building_class'].str.contains("District 5") == False]
bld_class2 = newdf['building_class'].unique()
print("Building class (clean): ", bld_class2)
print(newdf.head(10))

# Convert categorical variables to numeric using dummy variables

# Convert 'building_class' to numeric using get_dummies:
class_data = pd.get_dummies(newdf['building_class'], drop_first=True)
# Check to see dummy variable has been created
print(class_data.head(10))

# Concatenate dummy variables into original dataframe:
newdf = pd.concat([newdf, class_data], axis=1)
print(newdf.head(10))

# Drop original categorical variables from dataframe
newdf.drop(['building_class'], axis=1, inplace=True)
print(newdf.head(10))

# View column titles
print(newdf.columns)


# FEATURE SELECTION
# Correlogram: Visualise correlation of all variables with a heat map
cor = newdf.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(cor, cmap=plt.cm.CMRmap_r, annot=True)
plt.title("Heatmap Showing Correlation of Transactions Dataset")
plt.show()

# Correlation Analysis of ''Premium Grade A, Prime CBD', 'net_effective_rr_$/sqm'
print("Correlation between ''Premium Grade A, Prime CBD'' and 'net_effective_rr_$/sqm': ",
      newdf['Premium Grade A, Prime CBD'].corr(newdf['net_effective_rr_$/sqm']))

# Correlation Analysis of 'sub_sector_vacancy_%','net_effective_rr_$/sqm'
print("Correlation between 'sub_sector_vacancy_%' and 'net_effective_rr_$/sqm': ",
      newdf['sub_sector_vacancy_%'].corr(newdf['net_effective_rr_$/sqm']))

# Correlation Analysis of 'market_vacancy_%','net_effective_rr_$/sqm'
print("Correlation between 'market_vacancy_%' and 'net_effective_rr_$/sqm': ",
      newdf['market_vacancy_%'].corr(newdf['net_effective_rr_$/sqm']))

# Scatter plot with regression line of variables 'Premium Grade A, Prime CBD' and 'net_effective_rr_$/sqm' to visualise correlation
sns.lmplot(x='Premium Grade A, Prime CBD', y='net_effective_rr_$/sqm', data=newdf, x_jitter=0.1,
           y_jitter=0.1, line_kws={'color': 'red'})
plt.title('Scatter plot showing correlation between Premium Grade A, District 1 (CBD) and net_effective_rr_$/sqm')
plt.xlabel('Premium Grade A, Prime CBD')
plt.ylabel('net_effective_rr_$/sqm')
plt.show()

# Scatter plot with regression line of variables 'sub_sector_vacancy_%' and 'net_effective_rr_$/sqm' to visualise correlation
sns.lmplot(x='sub_sector_vacancy_%', y='net_effective_rr_$/sqm', data=newdf, x_jitter=0.1,
           y_jitter=0.1, line_kws={'color': 'red'})
plt.title('Scatter plot showing correlation between sub_sector_vacancy_% and net_effective_rr_$/sqm')
plt.xlabel('sub_sector_vacancy_%')
plt.ylabel('net_effective_rr_$/sqm')
plt.show()

# Scatter plot with regression line of variables 'market_vacancy_%' and 'net_effective_rr_$/sqm' to visualise correlation
sns.lmplot(x='market_vacancy_%', y='net_effective_rr_$/sqm', data=newdf, x_jitter=0.1,
           y_jitter=0.1, line_kws={'color': 'red'})
plt.title('Scatter plot showing correlation between market_vacancy_% and net_effective_rr_$/sqm')
plt.xlabel('market_vacancy_%')
plt.ylabel('net_effective_rr_$/sqm')
plt.show()

# Linear Regression Model:

# Setting the value for X and Y
X = newdf[['Premium Grade A, Prime CBD',
           'sub_sector_vacancy_%', 'market_vacancy_%']]
y = newdf['net_effective_rr_$/sqm']

# Multicollinearity check - VIF:
# compute the vif for all given features


def compute_vif(considered_features):

    X = newdf[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable'] != 'intercept']
    return vif


# features to consider removing
considered_features = ['Premium Grade A, Prime CBD',
                       'sub_sector_vacancy_%', 'market_vacancy_%']
# compute vif
print(compute_vif(considered_features).sort_values('VIF', ascending=False))

# Build Model
regr = linear_model.LinearRegression()
model = regr.fit(X, y)
model

# SCENARIO 1: OPTIMISTIC prediction of transacting rental rate (US$/sqm/annum):
print("SCENARIO 1: OPTIMISTIC prediction of transacting rental rate (US$/sqm/annum): ")
warnings.filterwarnings('ignore')
# Q2 2020:
predicted_trans_rr = regr.predict([[1, 14.8, 11.6]])
print("Predicted Transacting Rental Rate in Q2 2020 (US$/sqm/annum): ",
      predicted_trans_rr)
# Q3 2020:
predicted_trans_rr = regr.predict([[1, 13.6, 13.4]])
print("Predicted Transacting Rental Rate in Q3 2020 (US$/sqm/annum): ",
      predicted_trans_rr)
# Q4 2020:
predicted_trans_rr = regr.predict([[1, 15.5, 15.7]])
print("Predicted Transacting Rental Rate in Q4 2020 (US$/sqm/annum): ",
      predicted_trans_rr)
# Q1 2021:
predicted_trans_rr = regr.predict([[1, 12.8, 9.1]])
print("Predicted Transacting Rental Rate in Q1 2021 (US$/sqm/annum): ",
      predicted_trans_rr)


# SCENARIO 2: PESSIMISTIC prediction of transacting rental rate (US$/sqm/annum):
print("SCENARIO 2: PESSIMISTIC prediction of transacting rental rate (US$/sqm/annum): ")
warnings.filterwarnings('ignore')
# Q2 2020:
predicted_trans_rr = regr.predict([[1, 16.1, 12.8]])
print("Predicted Transacting Rental Rate in Q2 2020 (US$/sqm/annum): ",
      predicted_trans_rr)
# Q3 2020:
predicted_trans_rr = regr.predict([[1, 16.3, 13.7]])
print("Predicted Transacting Rental Rate in Q3 2020 (US$/sqm/annum): ",
      predicted_trans_rr)
# Q4 2020:
predicted_trans_rr = regr.predict([[1, 20.5, 17.3]])
print("Predicted Transacting Rental Rate in Q4 2020 (US$/sqm/annum): ",
      predicted_trans_rr)
# Q1 2021:
predicted_trans_rr = regr.predict([[1, 19.5, 10.0]])
print("Predicted Transacting Rental Rate in Q1 2021 (US$/sqm/annum): ",
      predicted_trans_rr)

# SCENARIO 3: Mean Average prediction of transacting rental rate (US$/sqm/annum):
print("SCENARIO 3: Mean Average  prediction of transacting rental rate (US$/sqm/annum): ")
warnings.filterwarnings('ignore')
# Q2 2020:
predicted_trans_rr = regr.predict([[1, 15.4, 12.2]])
print("Predicted Transacting Rental Rate in Q2 2020 (US$/sqm/annum): ",
      predicted_trans_rr)
# Q3 2020:
predicted_trans_rr = regr.predict([[1, 14.9, 14.1]])
print("Predicted Transacting Rental Rate in Q3 2020 (US$/sqm/annum): ",
      predicted_trans_rr)
# Q4 2020:
predicted_trans_rr = regr.predict([[1, 18.0, 16.5]])
print("Predicted Transacting Rental Rate in Q4 2020 (US$/sqm/annum): ",
      predicted_trans_rr)
# Q1 2021:
predicted_trans_rr = regr.predict([[1, 16.1, 9.6]])
print("Predicted Transacting Rental Rate in Q1 2021 (US$/sqm/annum): ",
      predicted_trans_rr)
