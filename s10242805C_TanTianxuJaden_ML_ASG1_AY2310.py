#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the required packages
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore


# ## 1. HR Analytics

# ### 1.1 Load and Explore the data

# In[2]:


# Load the full dataset
df_data = pd.read_csv('hr_data.csv')


# In[3]:


df_data.info()


# In[4]:


# numeric data
df_data_num = df_data.select_dtypes(['int64', 'float64']).copy()
df_data_num.head()


# In[5]:


# categorical data
df_data_cat = df_data.select_dtypes(['object']).copy()
df_data_cat.head()                       


# In[6]:


# Set the style for better aesthetics
sns.set(style="whitegrid")

# Select numerical and binary features along with the target variable
features = ['is_promoted', 'age', 'previous_year_rating', 'length_of_service', 'avg_training_score']
df_selected_features = df_data[features]

# Create side-by-side boxplots for numerical and binary features
plt.figure(figsize=(16, 8))

# Boxplots for numerical features and binary features
for i, feature in enumerate(features[1:]):  # Exclude 'is_promoted'l
    plt.subplot(2, 4, i+1)
    sns.boxplot(x='is_promoted', y=feature, data=df_selected_features)
    plt.title(f'Boxplot of {feature} by Promotion Status')

plt.tight_layout()
plt.show()


# ### 1.2 Cleanse and Transform the data

# ### 1.2.1 Missing Values

# In[7]:


df_data.isnull().sum()


# In[8]:


# Cleaning "Education" Column
df_data['education'].fillna('Unknown', inplace=True)  # Fill missing values with 'Unknown'
df_data['education'] = df_data['education'].str.lower()  # Standardize categories to lowercase

# Cleaning "Previous Year Rating" Column
df_data['previous_year_rating'].fillna(df_data['previous_year_rating'].median(), inplace=True)  # Impute missing values with the median
df_data['previous_year_rating'] = df_data['previous_year_rating'].astype(int)  # Ensure correct data type


# In[9]:


df_data.isnull().sum()


# #### Outliers

# In[10]:


# Assuming df_data is your DataFrame
age_column = df_data['age']
length_of_service_column = df_data['length_of_service']

# Function to calculate outliers
def count_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((column < lower_bound) | (column > upper_bound)).sum()

# Count outliers for age
outliers_count_age = count_outliers(age_column)

# Count outliers for length_of_service
outliers_count_length_of_service = count_outliers(length_of_service_column)

# Total number of rows
total_rows = len(df_data)

# Print the results
print(f"Number of outliers in 'age': {outliers_count_age}")
print(f"Number of outliers in 'length_of_service': {outliers_count_length_of_service}")
print(f"Total number of rows: {total_rows}")
print(f"Percentage of outliers in 'age': {100 * outliers_count_age / total_rows:.2f}%")
print(f"Percentage of outliers in 'length_of_service': {100 * outliers_count_length_of_service / total_rows:.2f}%")


# In[11]:


# Assuming df_data is your DataFrame
# For 'age'
Q1_age = df_data['age'].quantile(0.25)
Q3_age = df_data['age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
lower_bound_age = Q1_age - 1.5 * IQR_age
upper_bound_age = Q3_age + 1.5 * IQR_age

# Remove rows with outliers in 'age'
df_data = df_data[(df_data['age'] >= lower_bound_age) & (df_data['age'] <= upper_bound_age)]

# For 'length_of_service'
Q1_service = df_data['length_of_service'].quantile(0.25)
Q3_service = df_data['length_of_service'].quantile(0.75)
IQR_service = Q3_service - Q1_service
lower_bound_service = Q1_service - 1.5 * IQR_service
upper_bound_service = Q3_service + 1.5 * IQR_service

# Remove rows with outliers in 'length_of_service'
df_data = df_data[(df_data['length_of_service'] >= lower_bound_service) & (df_data['length_of_service'] <= upper_bound_service)]

df_data.describe()


# ### 1.2.2 Feature Engineering

# #### Drop the irrelevant features/ columns

# In[12]:


# Drop the 'employee_id','region','gender','department','recruitment_channel' columns
df_data = df_data.drop(['employee_id','region','gender'], axis = 1)
df_data.head()


# #### Create New Features

# In[13]:


# Create new feature
df_data['training_effectiveness'] = df_data['avg_training_score'] / df_data['no_of_trainings']

# Display the first few rows of the DataFrame
df_data.head()


# ### 1.2.3 Data transformation

# #### Encode Categorical Data Columns

# In[14]:


unique = df_data['education'].unique()
unique


# In[15]:


unique = df_data['department'].unique()
unique


# In[16]:


unique = df_data['recruitment_channel'].unique()
unique


# In[17]:


# education
df_data['education'] = df_data['education'].map( {"bachelor's": 0, 'below secondary': 1,"master's & above": 2,"unknown": 3} ).astype(int)
    


# In[18]:


# recruitment_channel
df_data['recruitment_channel'] = df_data['recruitment_channel'].map( {"sourcing": 0, 'other': 1,"referred": 2} ).astype(int)


# In[19]:


department_mapping = {department: idx for idx, department in enumerate(df_data['department'].unique())}

# Apply the mapping to encode the 'department' column
df_data['department'] = df_data['department'].map(department_mapping).astype(int)



# In[20]:


df_data.head()


# #### Transform the Numeric Data Columns

# In[21]:


df_data.describe()  


# In[22]:


import numpy as np

# Define conditions and corresponding values
conditions = [
    (df_data['training_effectiveness'] <= 20),
    (df_data['training_effectiveness'] > 20) & (df_data['training_effectiveness'] <= 40),
    (df_data['training_effectiveness'] > 40) & (df_data['training_effectiveness'] <= 60),
    (df_data['training_effectiveness'] > 60) & (df_data['training_effectiveness'] <= 80),
    (df_data['training_effectiveness'] > 80)
]

values = [0, 1, 2, 3, 4]

# Create a new column based on conditions
df_data['training_effectiveness_level'] = np.select(conditions, values, default=5)

# Convert the new column to integer type
df_data['training_effectiveness_level'] = df_data['training_effectiveness_level'].astype(int)


# In[23]:


df_data.head()


# In[24]:


effectiveness_counts = df_data['training_effectiveness_level'].value_counts()
print(effectiveness_counts)


# ### 1.3 Correlation Analysis

# In[25]:


df_data.corr()


# In[26]:


correlation_matrix = df_data.corr()
correlation_with_target = correlation_matrix['is_promoted'].sort_values(ascending=False)
correlation_with_target


# In[27]:


# Drop the weak correlation columns
df_data = df_data.drop(['education','recruitment_channel','length_of_service','department','age','no_of_trainings'], axis = 1)
df_data.head()


# In[28]:


# Heatmap: the correlation between any two features/variables
colormap = plt.cm.viridis
plt.figure(figsize=(10,10))
plt.title('Correlation of Features', size=15)
ax = sns.heatmap(df_data.astype(float).corr(), cmap=colormap, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# ### 1.4 Export the data

# In[29]:


df_data.to_csv('hr_data_new.csv', index=False)


# ## 2. Airbnb

# ### 2.1 Load and Explore the data

# In[30]:


df_listings = pd.read_csv('listings.csv')


# In[31]:


# numeric data
df_num_listings = df_listings.select_dtypes(['int64', 'float64']).copy()
df_num_listings.head()


# In[32]:


# categorical data
df_cat_listings = df_listings.select_dtypes(['object']).copy()
df_cat_listings.head()


# In[33]:


# Set the style for better aesthetics
sns.set(style="whitegrid")

# Select numerical and binary features along with the target variable
features = ['price','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']
df_selected_features = df_listings[features]

# Create side-by-side boxplots for numerical features and binary featuresS
plt.figure(figsize=(16, 8))

# Boxplots for numerical features
for i, feature in enumerate(features):
    plt.subplot(2, 4, i+1)
    sns.boxplot(x='price', y=feature, data=df_selected_features)
    plt.title(f'Boxplot of {feature} by Promotion Status')

plt.tight_layout()
plt.show()


# ### 2.2 Cleanse and Transform the data

# In[34]:


df_listings.isnull().sum()


# In[35]:


# Check if rows with null values are the same
null_rows_last_review = df_listings[df_listings['last_review'].isnull()]
null_rows_reviews_per_month = df_listings[df_listings['reviews_per_month'].isnull()]

# Check if the rows are the same
rows_are_same = null_rows_last_review.index.equals(null_rows_reviews_per_month.index)

print(f"Are the rows with null values the same between 'last_review' and 'reviews_per_month'? {rows_are_same}")


# In[36]:


# Cleaning "last_review" Column
df_listings['last_review'].fillna(0, inplace=True)  # Fill missing values with 0


# In[37]:


# Cleaning "reviews_per_month" Column
df_listings['reviews_per_month'].fillna(0, inplace=True)  # Fill missing values with 0


# In[38]:


df_listings.isnull().sum()


# #### Outliers

# In[39]:


# Assuming df_data is your DataFrame
number_of_reviews_column = df_listings['number_of_reviews']
reviews_per_month_column = df_listings['reviews_per_month']

# Function to calculate outliers
def count_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((column < lower_bound) | (column > upper_bound)).sum()

# Count outliers for outliers_count_number_of_reviews
outliers_count_number_of_reviews = count_outliers(number_of_reviews_column)

# Count outliers for reviews_per_month
outliers_count_reviews_per_month = count_outliers(reviews_per_month_column)

# Total number of rows
total_rows = len(df_listings)

# Print the results
print(f"Number of outliers in 'number_of_reviews': {outliers_count_number_of_reviews}")
print(f"Number of outliers in 'reviews_per_month': {outliers_count_reviews_per_month}")
print(f"Total number of rows: {total_rows}")
print(f"Percentage of outliers in 'number_of_reviews': {100 * outliers_count_number_of_reviews / total_rows:.2f}%")
print(f"Percentage of outliers in 'reviews_per_month': {100 * outliers_count_reviews_per_month / total_rows:.2f}%")


# In[40]:


number_of_reviews_column = df_listings['number_of_reviews']
reviews_per_month_column = df_listings['reviews_per_month']

# Function to remove outliers
def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df_listings[(column >= lower_bound) & (column <= upper_bound)]

# Remove outliers for 'number_of_reviews'
df_listings_no_outliers_number_of_reviews = remove_outliers(number_of_reviews_column)

# Remove outliers for 'reviews_per_month'
df_listings_no_outliers_reviews_per_month = remove_outliers(reviews_per_month_column)
df_listings.describe()


# ### 2.3 Correlation Analysis

# ### 2.2.2 Feature Engineering

# #### Drop the irrelevant features/ columns

# In[41]:


# Drop the 'id','name','host_id','host_name','calculated_host_listings_count' columns
df_listings = df_listings.drop(['id','name','host_id','host_name','calculated_host_listings_count'], axis = 1)
df_listings.head()


# #### Create New Features

# In[42]:


df_listings['average_price_per_night'] = df_listings['price'] / df_listings['minimum_nights']


# In[43]:


df_listings['location_popularity_index'] = df_listings['availability_365'] / df_listings['number_of_reviews'] 
df_listings.head()


# ### 2.2.3 Data transformation

# #### Encode Categorical Data Columns

# In[44]:


unique = df_listings['neighbourhood_group'].unique()
unique


# In[45]:


unique = df_listings['neighbourhood'].unique()
unique


# In[46]:


unique = df_listings['room_type'].unique()
unique


# In[47]:


# Encoding "neighbourhood_group" column
neighbourhood_group_mapping = {'North Region': 0, 'Central Region': 1, 'East Region': 2, 'West Region': 3, 'North-East Region': 4}
df_listings['neighbourhood_group'] = df_listings['neighbourhood_group'].map(neighbourhood_group_mapping).astype(int)


# In[48]:


# Encoding "room_type" column
room_type_mapping = {'Private room': 0, 'Entire home/apt': 1, 'Shared room': 2}
df_listings['room_type'] = df_listings['room_type'].map(room_type_mapping).astype(int)


# In[49]:


from datetime import datetime

df_listings['last_review'] = pd.to_datetime(df_listings['last_review'], errors='coerce')

# Calculate the time difference in months and replace the existing 'last_review' column
current_date = datetime.now()
df_listings['last_review'] = ((current_date - df_listings['last_review']) / pd.Timedelta(days=30)).fillna(0).astype(int)

# Display the DataFrame with the updated 'last_review' column in months
df_listings



# In[50]:


# Create a mapping for encoding
neighbourhood_mapping = {neighbourhood: idx for idx, neighbourhood in enumerate(df_listings['neighbourhood'].unique())}

# Apply the mapping to encode the 'neighbourhood' column
df_listings['neighbourhood'] = df_listings['neighbourhood'].map(neighbourhood_mapping).astype(int)

df_listings.head()


# array(['Woodlands', 'Bukit Timah', 'Tampines', 'Bedok', 'Bukit Merah',
#        'Newton', 'Geylang', 'River Valley', 'Jurong West', 'Rochor',
#        'Queenstown', 'Serangoon', 'Marine Parade', 'Pasir Ris',
#        'Toa Payoh', 'Outram', 'Punggol', 'Tanglin', 'Hougang', 'Kallang',
#        'Novena', 'Downtown Core', 'Bukit Panjang', 'Singapore River',
#        'Orchard', 'Ang Mo Kio', 'Bukit Batok', 'Museum', 'Sembawang',
#        'Choa Chu Kang', 'Central Water Catchment', 'Sengkang', 'Clementi',
#        'Jurong East', 'Bishan', 'Yishun', 'Mandai', 'Southern Islands',
#        'Sungei Kadut', 'Western Water Catchment', 'Tuas', 'Marina South',
#        'Lim Chu Kang']
#        
# It goes in sequence, 0 is Woodlands and bukit timah is 1

# #### Transform the Numeric Data Columns

# In[51]:


df_listings.describe()


# In[52]:


# Set up subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))

# Price
axes[0, 0].hist(df_listings['price'], bins=50, range=(0, 10000), color='skyblue', edgecolor='black')
axes[0, 0].set_title('Price')

# Minimum Nights
axes[0, 1].hist(df_listings['minimum_nights'], bins=50, range=(0, 1000), color='salmon', edgecolor='black')
axes[0, 1].set_title('Minimum Nights')

# Number of Reviews
axes[1, 0].hist(df_listings['number_of_reviews'], bins=50,range=(0, 365), color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Number of Reviews')

# Reviews Per Month
axes[1, 1].hist(df_listings['reviews_per_month'], bins=50,range=(0, 100), color='gold', edgecolor='black')
axes[1, 1].set_title('Reviews Per Month')

# Availability 365
axes[2, 0].hist(df_listings['availability_365'], bins=50,range=(0, 365), color='lightcoral', edgecolor='black')
axes[2, 0].set_title('Availability 365')

# Location Popularity Index
axes[2, 1].hist(df_listings['location_popularity_index'], bins=50,range=(0, 10000), color='lightblue', edgecolor='black')
axes[2, 1].set_title('Location Popularity Index')

# Adjust layout
plt.tight_layout()
plt.show()


# In[53]:


# Bin / Group location_popularity_index values
df_listings.loc[df_listings['location_popularity_index'] <= 122, 'popularity_level'] = 0
df_listings.loc[(df_listings['location_popularity_index'] > 122) & (df_listings['location_popularity_index'] <= 244), 'popularity_level'] = 1
df_listings.loc[df_listings['location_popularity_index'] > 244, 'popularity_level'] = 2

# Cleaning "reviews_per_month" Column
df_listings['location_popularity_index'].fillna(3, inplace=True)  # Fill missing values with 3
df_listings['popularity_level'] = df_listings['popularity_level'].astype(float)  # Change to float type

# Convert the new column to integer type, treating NaN and inf as 3
df_listings['popularity_level'] = df_listings['popularity_level'].fillna(3).astype(int)


# - **Comprehensive Popularity Measure:** The index considers both total reviews and availability, providing a holistic measure of a listing's popularity.
# - **Normalized Representation:** Normalizing the index to availability ensures fair comparisons among listings with varying levels of accessibility.
# - **Quantitative Indicator:** Assigning numeric values (0, 1, 2) enhances interpretability, making it suitable for statistical analyses and machine learning models.
# - **Facilitates Decision-Making:** Useful for hosts and travelers alike, the index aids in assessing demand and popularity for informed decision-making.
# - **Simplified Analysis:** Numeric values streamline subsequent analytical tasks, making the dataset suitable for various data science applications.

# In[54]:


df_listings.head()


# In[55]:


popularity_counts = df_listings['popularity_level'].value_counts()
print(popularity_counts)


# ### 2.3 Correlation Analysis

# In[56]:


correlation_matrix = df_listings.corr()
correlation_with_target = correlation_matrix['price'].sort_values(ascending=False)
correlation_with_target


# In[57]:


df_northregion = df_listings[df_listings['neighbourhood_group'] == 0]
df_centralregion = df_listings[df_listings['neighbourhood_group'] == 1]
df_eastregion = df_listings[df_listings['neighbourhood_group'] == 2]
df_westregion = df_listings[df_listings['neighbourhood_group'] == 3]
df_northeastregion = df_listings[df_listings['neighbourhood_group'] == 4]


# In[58]:


# Drop the weak correlation columns
df_listings = df_listings.drop(['availability_365','location_popularity_index','minimum_nights','reviews_per_month','longitude','number_of_reviews','latitude','last_review','neighbourhood_group'], axis = 1)
df_listings.head()


# In[59]:


# Heatmap: the correlation between any two features/variables
colormap = plt.cm.viridis
plt.figure(figsize=(10,10))
plt.title('Pearson Correlation of Features', size=15)
ax = sns.heatmap(df_listings.astype(float).corr(), cmap=colormap, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[60]:


# # Set relevant columns for the scatter matrix
# selected_columns = ['price', 'neighbourhood_group', 'neighbourhood', 'room_type', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365', 'average_price_per_night', 'location_popularity_index', 'popularity_level']

# # Set style
# sns.set(style="ticks")

# # Create a scatter matrix
# scatter_matrix = sns.pairplot(df_listings[selected_columns], hue='popularity_level', markers=["o", "s"], palette="husl")
# plt.suptitle("Scatter Matrix for Selected Columns", y=1.02, size=16)
# plt.show()


# In[ ]:





# ### 2.4 Export the data

# In[61]:


df_listings.to_csv('listings_new.csv', index=False)

