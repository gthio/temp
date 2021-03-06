# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%



# %%
#https://medium.com/@datalesdatales/why-you-should-be-plotting-learning-curves-in-your-next-machine-learning-project-221bae60c53

# %% [markdown]
#  # Problem formulation
#  ---
# 
#  This notebook contains detail work on building machine learning model(s) to predict house's sale price in Ames, Iowa. The model is build based on historical sale price with its 79 explanatory attributes like house area, lot size, house's condition, etc.
# 
# %% [markdown]
#  # Context
#  ---
#  Dataset covers house sale price in Ames - Iowa, from January 2006 to July 2010, with its 79 explanatory attributes describing every feature every feature of homes.
# 
# 
#  **Data sources**
#  * **train.csv** - training dataset
#  * **test.csv** - test dataset
#  * **data_description.txt** - dataset's metadata
# 
# 
# %% [markdown]
#  # Development environment setup
#  ---
# 
#  1. Import packages
#      * numpy, scipy, pandas, matplotlib + seaborn
#      * sklearn
#  2. Set common configurations
#      * dataframe's options
#      * matplotlib, seaborn style

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as style
import seaborn as sns
from datetime import datetime

from scipy import stats

# sklearn's pre-processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# sklearn's feature selection
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

# sklearn's model and metrics
from sklearn.model_selection import train_test_split, cross_validate, learning_curve, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# %%



# %%
# set dataframe display options
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 100)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
    
#set pyplot and seaborn style
sns.set(style="whitegrid", palette="deep", color_codes=True)
style.use('fivethirtyeight')

# set seed for reproducibility
np.random.seed(0) 

# %% [markdown]
#  # Common custom functions
#  ---

# %%
# get dataframe meta data

def get_meta_data(df):
    
    # get column's datatype
    temp1 = pd.DataFrame({'dtype': df.dtypes}).T

    # get column's description
    temp2 = df.describe(include='all')

    # combine all
    temp = pd.concat([temp1, temp2]).T

    return temp


# %%
# get missing data info for each attribute

def get_stat_null(x, df_data):
    
    col = x.name
    count_null = df_data[col].isnull().sum()
    count_notnull = df_data[col].notnull().sum()
    pct_null = count_null / (count_null + count_notnull)

    return [count_notnull, count_null, pct_null]


# %%
# get each attribute's uniqueness data

def get_stat_uniqueness(x, df_data):
    
    col = x.name
    count = len(df_data[col].dropna())
    uniques = df_data[col].dropna().unique()
    
    flatten = '...'
    unique_count = len(uniques) #float('nan')
    
    if (unique_count < 16):
        flatten = ', '.join([str(x) for x in uniques if ((x is not ''))])
        unique_count = len(uniques)

    return [unique_count, flatten]


# %%
# get each attribute's statistic value (ie. skew and kurtosis)

def get_stat(x, df_data):
    
    col = x.name
    
    skewness = float('nan')
    kurtosis = float('nan')

    if ((x['dtype'] == 'int64') | (x['dtype'] == 'float64')):
        skewness = df_data[col].skew()
        kurtosis = df_data[col].kurtosis()
        
    return [skewness, kurtosis]


# %%
# get dataframe stat (metadata)

def get_dataframe_metadata(df):
    
    # get and set dataframe's metadata
    df_meta = get_meta_data(df)

    # set stat info for each attribute
    set_stat(df, df_meta)

    return df_meta


# %%
# get each attribute's 3rd, 2nd and 1st quartile deviation

def get_stat_quart(x):
    
    col = x.name
    
    upper1 = float('nan'); lower1 = float('nan')
    upper2 = float('nan'); lower2 = float('nan')
    upper3 = float('nan'); lower3 = float('nan')    

    if ((x['dtype'] == 'int64') | (x['dtype'] == 'float64')):
        mean = x['mean']
        stddev = x['std']

        lower3 = mean - (stddev * 3); upper3 = mean + (stddev * 3)
        lower2 = mean - (stddev * 2); upper2 = mean + (stddev * 2)        
        lower1 = mean - (stddev * 1); upper1 = mean + (stddev * 1)        
        
    return [lower1, upper1, lower2, upper2, lower3, upper3]


# %%
# set each attribute's stats

def set_stat(df_data, df_meta):
    
    # set missing data info for each attribute
    df_meta[['notnull_count', 'null_count', 'null_pct']] = df_meta.loc[:, :].apply(
        lambda x: get_stat_null(x, df_data), axis=1, result_type="expand")

    # set each attribute's uniqueness data
    df_meta[['unique_count', 'unique_values']] = df_meta.loc[:, :].apply(
        lambda x: get_stat_uniqueness(x, df_data), axis=1, result_type="expand")

    # set each attribute's statistic value (ie. skew and kurtosis)
    df_meta[['skew', 'kurtosis']] = df_meta.loc[:, :].apply(
        lambda x: get_stat(x, df_data), axis=1, result_type="expand")

    # set each attribute's 3rd, 2nd and 1st quartile deviation
    df_meta[['lower_3s_1', 'upper_3s_1', 
             'lower_3s_2', 'upper_3s_2', 
             'lower_3s_3', 'upper_3s_3']] = df_meta.loc[:, :].apply(
        lambda x: get_stat_quart(x), axis=1, result_type="expand")
    


# %%
# compute correlation

def compute_correlation_matrix(df, target, attributes):
    
    # compute correlation matrix
    df_corr = df[target+attributes].corr()
    
    # order by correlation to target, descending 
    temp1 = df_corr.iloc[:, :].sort_values(by=target, ascending=False)
    
    # get features list - to reorder columns
    temp2 = temp1.iloc[:, 0].index[:]
    
    result = temp1.loc[:, temp2]
    
    return result


# %%
# plot pca - 1D

def plot_pca(df, target, attributes):
    
    target_string = target[0]
    
    pca = PCA(n_components=1)
    pca.fit(df[attributes])
    cols_1d = pca.transform(df[attributes])
    
    fig, ax = plt.subplots()
    ax.scatter(cols_1d, df[target])
    ax.set_title('1D PCA plot (selected ' + str(len(attributes)) + ' attributes)')
    ax.set_xlabel('cols 1d')
    ax.set_ylabel(target_string)
    plt.show()
    
    x = pd.DataFrame({target_string: df[target_string], 'cols_1d' : cols_1d.flatten()}).corr()
    
    print(x)    
    
    return cols_1d


# %%
# plot target to features 1-d pca

def plot_pca_smarter(df, df_meta, target, number_of_attributes):


    attributes = df_meta.loc[(np.in1d(df_meta.dtype, dtype_numeric)) & (df_meta['null_count'] == 0)][1:number_of_attributes].index

    cols_1d = plot_pca(df, target, attributes)
    
def plot_pca_smarter2(df, df_meta, target, attributes):

    cols_1d = plot_pca(df, target, attributes)
    


# %%
# plot correlations

def plot_correlation_matrix(df_corr):
    
    mask = np.zeros_like(df_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    f, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(df_corr, linewidths=.10, annot=False, mask=mask, fmt='.2g', cmap="PiYG");
    


# %%
# plot attribute charts

def plot_attribute_chart(df, attribute):

    # create chart
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # histogram
    ax1 = fig.add_subplot(grid[0, :2])
    sns.distplot(df.loc[:, attribute], norm_hist=True, ax=ax1)

    # qq plot
    ax2 = fig.add_subplot(grid[1, :2])
    stats.probplot(df.loc[:, attribute], plot=ax2)

    # boxplot
    ax3 = fig.add_subplot(grid[0:3, 2])
    sns.boxplot(df.loc[:, attribute], orient='v', ax=ax3)

    #https://matplotlib.org/mpl-probscale/tutorial/closer_look_at_viz.html


# %%
# plot attribute charts - multiple

def plot_attributes_chart(df, attributes):

    for attribute in attributes:
        plot_attribute_chart(df, attribute)


# %%
# get df meta data, order by correlation to target

def get_df_meta_by_correlation(df_corr, df_meta, target):
    
    column_name = target + '_Corr'
    
    df_temp = df_corr.copy(deep=True)
    
    df_temp = df_temp[[target]].rename(
        columns={target: column_name})

    result = pd.merge(df_temp, 
        df_meta, 
        left_index=True, 
        right_index=True, 
        how='inner').sort_values(by=column_name, ascending=False)
    
    return result

def get_df_meta_by_correlation2(df_corr, df_meta, target):
    
    column_name = target + '_Corr'
    column_name_abs = target + '_Corr_Abs'
    
    df_temp = df_corr.copy(deep=True)
        
    df_temp = df_temp[[target]].rename(
        columns={target: column_name})

    df_temp[column_name_abs] = df_temp[column_name].abs()    
    
    result = pd.merge(df_temp, 
        df_meta, 
        left_index=True, 
        right_index=True, 
        how='inner').sort_values(by=column_name_abs, ascending=False)
    
    return result


# %%
# impute missing data

def set_missing_data_with_freq_value(df, attribute):
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer.fit(df[[attribute]])
    
    df[attribute] = imputer.transform(df[[attribute]])

def set_missing_data_with_value(df, attribute, fill_value):
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=fill_value)
    imputer.fit(df[[attribute]])
    
    df[attribute] = imputer.transform(df[[attribute]])


# %%
def plot_attribute_categorical(df, target, attribute):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    sns.catplot(x=attribute, y=target, kind='bar', data=df_raw_train.fillna('MISSING!!'), ax=ax[0])
    sns.countplot(df_raw_train[attribute].fillna('MISSING!!'), ax=ax[1])
    plt.close(2)
    plt.show()


# %%


# %% [markdown]
#  # Data engineering
#  -----
#  1. Load data
#  2. Briefly check both train and test dataset
# 
# 
# 
# 
# 

# %%
# set data path and file names
file_path = "/kaggle/input/house-prices-advanced-regression-techniques/"
file_path = ""
file_name_train = "train.csv"
file_name_test = "test.csv"
file_name_meta = "data_description.txt"
file_name_submission = "sample_submission.csv"
file_separator = ","

# these are numeric data types
dtype_numeric = ['int64', 'uint8', 'float64']

# total number of rows to show
config_df_row_count = 8
config_df_row_correlation_count = 20

# target variable
target = ['SalePrice']

# variable - ignored list
variable_ignored = ['Id']

# variable - year month list
variable_year_month = ['YearBuilt', 'YearRemodAdd', 'YrSold', 'MoSold']


# %%
# read in data
df_raw_train = pd.read_csv(
    file_path+file_name_train, 
    sep=file_separator)

df_raw_test = pd.read_csv(
    file_path+file_name_test, 
    sep=file_separator)


# %%


# %% [markdown]
#  ## Data - as it is
#  ---
# 
#  1. The dataset contains 1460 training set and 1459 test set.
#  2. The training set has the sale price while the test set does not.
#  3. There are 33 numeric attributes and 46 categorical attributes (23 nominal, 23 ordinal)
# 
# 
# 
# 

# %%
# check for data's shape
file_names = [file_name_train, file_name_test]
row_counts = [df_raw_train.shape[0], df_raw_test.shape[0]]
col_counts = [df_raw_train.shape[1], df_raw_test.shape[1]]
list_of_tuples = list(zip(file_names, row_counts, col_counts))  
    
pd.DataFrame(list_of_tuples, columns = ['file_name', 'row_counts', 'col_counts'])  


# %%
# sample train data
df_raw_train.sample(config_df_row_count)


# %%
# sample test data
df_raw_test.head(config_df_row_count)


# %%
# get train data metadata
df_raw_train_meta = get_dataframe_metadata(df_raw_train)
df_raw_train_meta.head(config_df_row_count)


# %%
# get test data metadata
df_raw_test_meta = get_dataframe_metadata(df_raw_test)
df_raw_test_meta.head(config_df_row_count)


# %%


# %% [markdown]
#  ### SalePrice
#  ---
#  1. SalePrice is the target variable
#  2. Not normally distributed (skew: 1.88, kurtosis: 6.54)
#  3. Has some outliers
# 

# %%
attribute = 'SalePrice'
plot_attribute_chart(df_raw_train, attribute)
df_raw_train_meta.loc[df_raw_train_meta.index == attribute, :]


# %%


# %% [markdown]
#  ### Exterior
#  ---
# 
#  1. **ExterCond**, evaluates the present condition of the material on the exterior (**Ex**: Excellent, **Gd**: Good, **TA**: Average/Typical, **Fa**: Fair, **Po**:	Poor)
# 
#  2. **ExterQual**, evaluates the quality of the material on the exterior
# 
#  3. **Exterior1st** , **Exterior2nd**: Exterior covering on house
# 
#  4. **Fence quality** (**GdPrv**: Good Privacy, **MnPrv**: Minimum Privacy, **GdWo**: Good Wood, **MnWw**: Minimum Wood/Wire, **NA**: No Fence)
# 
#  5. **RoofMatl**: Roof material (**ClyTile**: Clay or Tile, **CompShg**: Standard (Composite) Shingle, **Membran**: Membrane, **Metal**: Metal, **Roll**: Roll, **Tar&Grv**: Gravel & Tar, **WdShake**: Wood Shakes, **WdShngl**: Wood Shingles)
# 
#  6. **RoofStyle**: Type of roof (**Flat**: Flat, **Gable**: Gable, **Gambrel**: Gabrel (Barn), **Hip**: Hip, **Mansard**: Mansard, **Shed**: Shed)

# %%
plot_attribute_categorical(df_raw_train, target[0], 'ExterCond')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'ExterQual')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Exterior1st')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Exterior2nd')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Fence')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'RoofMatl')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'RoofStyle')


# %%


# %% [markdown]
#  ### Garage
#  ---
# 
#  1. **GarageArea**, size of garage in square feet
#  2. **GarageCars**, size of garage in car capacity
#  3. **GarageQual**, garage quality
#  4. **GarageCond**, garage condition
#  5. **GarageFinish**, interior finish of the garage
#  6. **GarageType**, garage location
#  7. **GarageYrBlt**, year garage was built

# %%
sns.regplot(x="GarageArea", y="SalePrice", data=df_raw_train);


# %%
plot_attribute_categorical(df_raw_train, target[0], 'GarageCars')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'GarageQual')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'GarageCond')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'GarageFinish')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'GarageType')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'GarageYrBlt')


# %%


# %% [markdown]
#  ### Basement
#  ---
#  1. **TotalBsmtSF**, total square feet of basement area
#  2. **BsmtQual**, evaluates the height of the basement
#  3. **BsmtCond**, evaluates the general condition of the basement
#  4. **BsmtExposure**, refers to walkout or garden level walls
#  5. **BsmtFinType1** , **BsmtFinType1**, rating of basement finished area
#  6. **BsmtFinSF1**, type 1 finished square feet
#  7. **BsmtFinSF2**, type 2 finished square feet
#  7. **BsmtUnfSF**, Unfinished square feet of basement area
# 

# %%
sns.regplot(x="TotalBsmtSF", y="SalePrice", data=df_raw_train);


# %%
plot_attribute_categorical(df_raw_train, target[0], 'BsmtQual')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'BsmtCond')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'BsmtExposure')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'BsmtFinType1')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'BsmtFinType2')


# %%
sns.regplot(x="BsmtFinSF1", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="BsmtFinSF2", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="BsmtUnfSF", y="SalePrice", data=df_raw_train);


# %%


# %% [markdown]
#  ### Kitchen
#  ---
#  1. **Kitchen**, kitchens above grade
#  2. **KitchenQual**, kitchen quality

# %%
plot_attribute_categorical(df_raw_train, target[0], 'KitchenAbvGr')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'KitchenQual')


# %%


# %% [markdown]
#  ### Living area
#  ---
#  1. **Fireplaces**, number of fireplaces
#  2. **FireplaceQu**, fireplace quality
#  3. **GrLivArea**, Above grade (ground) living area square feet

# %%
plot_attribute_categorical(df_raw_train, target[0], 'Fireplaces')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'FireplaceQu')


# %%
sns.regplot(x="GrLivArea", y="SalePrice", data=df_raw_train);


# %%


# %% [markdown]
#  ### Bedrooms and bathrooms
#  ---
#  1. **BedroomAbvGr**, bedrooms above grade (does NOT include basement bedrooms)
#  2. **TotRmsAbvGrd**, total rooms above grade (does not include bathrooms)
#  3. **FullBath**, full bathrooms above grade
#  4. **HalfBath**, half baths above grade
#  5. **BsmtFullBath**, basement full bathrooms
#  6. **BsmtHalfBath**, basement half bathrooms

# %%
plot_attribute_categorical(df_raw_train, target[0], 'BedroomAbvGr')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'TotRmsAbvGrd')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'FullBath')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'HalfBath')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'BsmtFullBath')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'BsmtHalfBath')


# %%


# %% [markdown]
#  ### Facilities
#  ---
#  1. **PoolArea**, pool area in square feet
#  2. **PoolQC**, pool quality
#  3. **MiscFeature**: Miscellaneous feature not covered in other categories
#  4. **MiscVal**: $Value of miscellaneous feature

# %%
plot_attribute_categorical(df_raw_train, target[0], 'PoolQC')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'PoolArea')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'MiscFeature')


# %%
sns.regplot(x="MiscVal", y="SalePrice", data=df_raw_train);


# %%


# %% [markdown]
#  ### Utilities
#  ---
#  1. **Electrical**: Electrical system
#  2. **CentralAir**: Central air conditioning
#  3. **Heating**: Type of heating
#  4. **HeatingQC**: Heating quality and condition
#  5. **Utilities**: Type of utilities available

# %%
plot_attribute_categorical(df_raw_train, target[0], 'Electrical')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'CentralAir')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Heating')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'HeatingQC')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Utilities')


# %%


# %% [markdown]
#  ### Building - general
#  ---
#  1. **OverallQual**, rates the overall material and finish of the house
#  2. **OverallCond**, rates the overall condition of the house

# %%
plot_attribute_categorical(df_raw_train, target[0], 'OverallQual')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'OverallCond')


# %%


# %% [markdown]
#  ### Building
#  ---
#  1. **BldgType**, type of dwelling
#  2. **HouseStyle**, style of dwelling
#  3. **Foundation**, type of foundation
#  4. **Functional**, home functionality (Assume typical unless deductions are warranted)
#  5. **MasVnrArea**, masonry veneer area in square feet
#  6. **MasVnrType**, masonry veneer type
#  7. **1stFlrSF**, first floor square feet
#  8. **2ndFlrSF**, second floor square feet
#  9. **LowQualFinSF**, low quality finished square feet (all floors)
#  10. **WoodDeckSF**, wood deck area in square feet
#  11. **OpenPorchSF**, open porch area in square feet
#  12. **EnclosedPorch**, enclosed porch area in square feet
#  13. **3SsnPorch**, three season porch area in square feet
#  14. **ScreenPorch**, screen porch area in square feet
#  15. **YearBuilt**, original construction date
#  16. **YearRemodAdd**, remodel date (same as construction date if no remodeling or additions)
# 

# %%
plot_attribute_categorical(df_raw_train, target[0], 'BldgType')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'HouseStyle')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Foundation')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Functional')


# %%
sns.regplot(x="MasVnrArea", y="SalePrice", data=df_raw_train);


# %%
plot_attribute_categorical(df_raw_train, target[0], 'MasVnrType')


# %%
sns.regplot(x="1stFlrSF", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="2ndFlrSF", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="LowQualFinSF", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="WoodDeckSF", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="OpenPorchSF", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="EnclosedPorch", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="3SsnPorch", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="ScreenPorch", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="YearBuilt", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="YearRemodAdd", y="SalePrice", data=df_raw_train);


# %%


# %% [markdown]
#  ### Lot
#  ---
#  1. **LotArea**, lot size in square feet
#  2. **LotFrontage**, linear feet of street connected to property
#  3. **LotShape**, general shape of property
#  4. **LotConfig**, lot configuration
# 

# %%
sns.regplot(x="LotArea", y="SalePrice", data=df_raw_train);


# %%
sns.regplot(x="LotFrontage", y="SalePrice", data=df_raw_train);


# %%
plot_attribute_categorical(df_raw_train, target[0], 'LotShape')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'LotConfig')


# %%


# %% [markdown]
#  ### Surroundings
# 
#  1. **Alley**, type of alley access to property
#  2. **PavedDrive**, paved driveway
#  3. **Street**, type of road access to property
#  4. **LandContour**, flatness of the property
#  5. **LandSlope**, slope of property
#  6. **Condition1**, proximity to various conditions
#  7. **Condition2**, proximity to various conditions (if more than one is present)

# %%
plot_attribute_categorical(df_raw_train, target[0], 'Alley')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'PavedDrive')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Street')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'LandContour')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'LandSlope')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Condition1')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Condition2')


# %%


# %% [markdown]
#  ### Area
#  ---
#  1. **MSZoning**, identifies the general zoning classification of the sale.
#  2. **Neighborhood**, physical locations within Ames city limits
# 

# %%
plot_attribute_categorical(df_raw_train, target[0], 'MSZoning')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'Neighborhood')


# %%


# %% [markdown]
#  ### Sell condition
#  ---
#  1. **SaleType**, type of sale
#  2. **SaleCondition**, condition of sale
#  3. **YrSold**, year sold (YYYY)
#  4. **MoSold**, month sold (MM)
# 
# 

# %%
plot_attribute_categorical(df_raw_train, target[0], 'SaleType')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'SaleCondition')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'YrSold')


# %%
plot_attribute_categorical(df_raw_train, target[0], 'MoSold')


# %%



# %%
#with pd.ExcelWriter('output.xlsx') as writer:
#    df_raw_train.to_excel(writer, sheet_name='train')
#    df_raw_train_meta.to_excel(writer, sheet_name='train_meta')

# %% [markdown]
#  ## Data as it is - numeric correlation

# %%
# get numeric attributes
numeric_attributes = df_raw_train_meta[np.in1d(df_raw_train_meta.dtype, dtype_numeric)].index
numeric_attributes = list(set(numeric_attributes) - set(variable_ignored) - set(variable_year_month) - set(target))


# %%
# compute correlation matrix
df_raw_train_corr_matrix = compute_correlation_matrix(
    df_raw_train, 
    target,
    numeric_attributes)

# plot correlations
plot_correlation_matrix(
    df_raw_train_corr_matrix)


# %%
# get attribute's description, order by correlation
df_raw_train_meta_sorted = get_df_meta_by_correlation2(
    df_raw_train_corr_matrix, 
    df_raw_train_meta,
    target[0])

df_raw_train_meta_sorted.head(
    config_df_row_correlation_count)


# %%
# plot target to features 1-d pca
plot_pca_smarter(df_raw_train, 
    df_raw_train_meta_sorted, 
    target, 
    14)


# %%
#https://seaborn.pydata.org/tutorial/regression.html


# %%


# %% [markdown]
#  ## Data - baseline cleaning
#  ---
# 
#  Based on the **data as it is** inspection,
# 
#  1. Derive atrribute from other attribute(s)
#      * YrMoSold --> YrSold + MoSold --> Yr*100 + Mo (X)
#      * Age --> (YrSold + MoSold) - YearBuilt
# 
#  2. Categorical attributes - one hot encoding
#      * BldgType: Type of dwelling
#      * HouseStyle: Style of dwelling
# 
#  3. Categorical attributes - specified order numeric encoding
#      * LotShape
#      * Street - Type of road access to property
#      * Alley - Type of alley access to property
#      * LandSlope - Slope of property
#      * HeatingQC - Heating quality and condition
#      * CentralAir - Central air conditioning
#      * KitchenQual - Kitchen quality
#      * GarageCond - Garage condition
#      * PoolQC -  Pool quality
# 
#  4.  Missing data handling
#      * MasVnrArea (Masonry veneer area in square feet) --> if Null set to 0
#      * LotFrontage (Linear feet of street connected to property) --> set to most frequent value
#      * GarageYrBlt --> set to YearBuilt if Null
# 
#  5. Subjectively remove columns that represent similar attribute
#      * Id (not relevant)
#      * GarageArea (highly correlated with GarageCars)
#      * GarageYrBlt (highly correlated with YearBuilt and YearRemodAdd)
#      * TotRmsAbvGrd (highly correlated with GrLivArea)
#      * BsmtFinSF1 and BsmtFinSF2 (equivalent to TotalBsmtSF)
# 
# 
# 

# %%
# clone previous df to this section df
df_base_train = df_raw_train.copy(deep=True)
df_base_test = df_raw_test.copy(deep=True)


# %%
# handler for missing data

def data_missing_handler(df):

    # set MasVnrArea 0 if its null
    set_missing_data_with_value(df, 'MasVnrArea', 0)

    # set MasVnrArea 0 if its null
    set_missing_data_with_value(df, 'GarageCars', 0)    

    # set MasVnrArea 0 if its null
    set_missing_data_with_value(df, 'FullBath', 0)
    set_missing_data_with_value(df, 'BsmtFullBath', 0)
    set_missing_data_with_value(df, 'HalfBath', 0)
    set_missing_data_with_value(df, 'BsmtHalfBath', 0)
    
    # set MasVnrArea 0 if its null
    set_missing_data_with_value(df, 'BsmtUnfSF', 0) 
    
    # set MasVnrArea 0 if its null
    set_missing_data_with_value(df, 'TotalBsmtSF', 0)
    set_missing_data_with_value(df, '1stFlrSF', 0)
    set_missing_data_with_value(df, '2ndFlrSF', 0)
    
    set_missing_data_with_value(df, 'OpenPorchSF', 0)    
    set_missing_data_with_value(df, 'EnclosedPorch', 0)    
    set_missing_data_with_value(df, '3SsnPorch', 0)    
    set_missing_data_with_value(df, 'ScreenPorch', 0)    
    set_missing_data_with_value(df, 'WoodDeckSF', 0)
    
    # set LotFrontage to most frequent value if its null
    #set_missing_data_with_freq_value(df, 'LotFrontage')
    
    #df['LotFrontage'] = df.apply(lambda x: (x['LotArea'] * 0.001) if x['LotFrontage'].isnull() else x['LotFrontage'])    
    
    df.loc[df['LotFrontage'].isnull(), 'LotFrontage'] = df.loc[df['LotFrontage'].isnull(), 'LotArea'] * 0.0025
    
    # set TotalBsmtSF to most frequent value if its null
    set_missing_data_with_freq_value(df, 'TotalBsmtSF')
   
    # set GarageYrBlt to YearBuilt if its null
    df.loc[df['GarageYrBlt'].isnull(), ['GarageYrBlt']] = df['YearBuilt']

data_missing_handler(df_base_train)
data_missing_handler(df_base_test)

# %% [markdown]
#  ### Derives attribute(s)

# %%
def data_derive_attributes(df):
    
    # age of the house
    df['d_HouseAge'] = (df['YrSold'] + (df['MoSold'] / 12)) - df['YearBuilt']

    # number of years since last renovation
    df['d_RemodAge'] = (df['YrSold'] + (df['MoSold'] / 12)) - df['YearRemodAdd']

    # just take the age as renovation age
    df['d_Age'] = df['d_RemodAge']
        
    df['d_HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)    
    df['d_HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    
    df['d_Bath'] = df['FullBath'] + df['BsmtFullBath'] + (0.5 * df['HalfBath']) + (0.5 * df['BsmtHalfBath'])
    df['d_TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF'] 
    # total floor sqf

    df['d_HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    
    
    df['d_HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df['d_Has2ndFlr'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    
    df['d_TotalFlrSF'] = df['1stFlrSF'] + df['2ndFlrSF']
    df['d_TotalBldgSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['d_GardenSF'] = df['LotArea'] - df['1stFlrSF']
    
data_derive_attributes(df_base_train) 
data_derive_attributes(df_base_test) 


# %%
# derive Neighborhood's code value based on the price per square feet

def data_derive_neighborhood_code(df):
    
    # get new dataframe for temp processing
    df_temp = df_base_train[['Neighborhood', 'SalePrice', 'd_TotalBldgSF']].copy(deep=True)

    # compute psf, price per sequare feet
    df_temp['d_PricePerSF'] = df_base_train['SalePrice'] / df_base_train['d_TotalBldgSF']

    # compute psf for each Neighborhood
    df_temp_group = df_temp.groupby(['Neighborhood'], as_index=False).agg({"d_PricePerSF": [np.mean, np.median]})
    df_temp_group.columns = ['_'.join(t).rstrip('_') for t in df_temp_group.columns]

    # sort dataframe on psf asncending order
    df_temp_group.sort_values(by="d_PricePerSF_median", ascending=True, inplace=True)

    # set computed Neighborhood's code value
    #df_temp_group['d_Neighborhood_Code'] = df_temp_group.reset_index().index + 1
    df_temp_group['d_Neighborhood_Code'] = df_temp_group['d_PricePerSF_median'] 
    
    return df_temp_group


df_neighborhood_code = data_derive_neighborhood_code(df_base_train)

df_base_train = pd.merge(df_base_train, 
    df_neighborhood_code[['Neighborhood', 'd_Neighborhood_Code']], 
    how='inner', 
    left_on='Neighborhood', 
    right_on='Neighborhood') 

df_base_test = pd.merge(df_base_test, 
    df_neighborhood_code[['Neighborhood', 'd_Neighborhood_Code']], 
    how='inner', 
    left_on='Neighborhood', 
    right_on='Neighborhood') 


# %%
# derive Neighborhood's code value based on the price per square feet

def data_derive_MSSubClass_code(df):
    
    # get new dataframe for temp processing
    df_temp = df_base_train[['MSSubClass', 'SalePrice', 'd_TotalBldgSF']].copy(deep=True)

    # compute psf, price per sequare feet
    df_temp['d_PricePerSF'] = df_base_train['SalePrice'] / df_base_train['d_TotalBldgSF']

    # compute psf for each Neighborhood
    df_temp_group = df_temp.groupby(['MSSubClass'], as_index=False).agg({"d_PricePerSF": [np.mean, np.median]})
    df_temp_group.columns = ['_'.join(t).rstrip('_') for t in df_temp_group.columns]

    # sort dataframe on psf asncending order
    df_temp_group.sort_values(by="d_PricePerSF_median", ascending=True, inplace=True)

    # set computed Neighborhood's code value
    df_temp_group['d_MSSubClass_Code'] = df_temp_group.reset_index().index + 1
    df_temp_group['d_MSSubClass_Code'] = df_temp_group['MSSubClass']
    
    return df_temp_group


df_MSSubClass_code = data_derive_MSSubClass_code(df_base_train)

dict_MSSubClass_code = df_MSSubClass_code[["MSSubClass", "d_MSSubClass_Code"]].set_index("MSSubClass")["d_MSSubClass_Code"].to_dict()

df_base_train["d_MSSubClass_Code"] = df_base_train["MSSubClass"].apply(lambda x: dict_MSSubClass_code.get(x, None))
df_base_test["d_MSSubClass_Code"] = df_base_test["MSSubClass"].apply(lambda x: dict_MSSubClass_code.get(x, None))


# %%
dict_MSSubClass_code

# %% [markdown]
#  #### One-hot encoding

# %%

def data_onehot_encoding(df):
    
    
    # one hot encoding - MSZoning
    #df = pd.concat([df, 
    #pd.get_dummies(df['MSZoning'], prefix='MSZoning')], axis=1) 
    
    # one hot encoding - BldgType
    #df = pd.concat([df, 
    #pd.get_dummies(df['BldgType'], prefix='BldgType')], axis=1)

    # one hot encoding - HouseStyle
    #df = pd.concat([df, 
    #pd.get_dummies(df['HouseStyle'], prefix='HouseStyle')], axis=1) 
    
    return df

df_base_train = data_onehot_encoding(df_base_train)
df_base_test = data_onehot_encoding(df_base_test)

df_base_train


# %%


# %% [markdown]
#  #### Numeric encoding

# %%
def data_numeric_encoding(df):

    # numeric encoding - GarageQual
    #df['OverallCond_Encoded'] = df['OverallCond'].map( 
    #    {10:4, 9:4, 8:4, 7:3, 6:3, 5:2, 4:2, 3:2, 2:1, 1:1})    
    
    
    # numeric encoding - GarageQual
    df['BsmtQual_Encoded'] = df['BsmtQual'].map( 
        {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, np.NaN:2})
    
    # numeric encoding - BsmtCond
    ##df['BsmtCond_Encoded'] = df['BsmtCond'].map( 
    ##    {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, np.NaN:2})    
    
    # numeric encoding - BsmtExposure
    df['BsmtExposure_Encoded'] = df['BsmtExposure'].map( 
        {'Ex':5, 'Gd':4, 'Av':3, 'Mn':2, 'No':1, np.NaN:1})         
    
    # numeric encoding - ExterQual
    df['ExterQual_Encoded'] = df['ExterQual'].map( 
        {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1})

    # numeric encoding - ExterCond
    df['ExterCond_Encoded'] = df['ExterCond'].map( 
        {'Ex':4, 'Gd':3, 'TA':3, 'Fa':2, 'Po':1})
       
    # numeric encoding - CentralAir
    df['CentralAir_Encoded'] = df['CentralAir'].map( 
        {'Y':2, 'N':1})   
    
    # numeric encoding - CentralAir
    df['Electrical_Encoded'] = df['Electrical'].map( 
        {'SBrkr':3, 'FuseA':2, 'FuseF':2, 'FuseP':2, 'Mix':1, np.NaN:3})       
    
    # numeric encoding - KitchenQual
    df['KitchenQual_Encoded'] = df['KitchenQual'].map( 
        {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1})   
    
    # numeric encoding - HeatingQC
    df['HeatingQC_Encoded'] = df['HeatingQC'].map( 
        {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1})   
    
    # numeric encoding - Alley
    df['Alley_Encoded'] = df['Alley'].map( 
        {'Pave':2, 'Grvl':1})    

    # numeric encoding - PoolQC
    #df['PoolQC_Encoded'] = df['PoolQC'].map( 
    #    {'Ex':4, 'Gd':3, 'Fa':2, np.NaN:1})   
    
    # numeric encoding - LotShape
    df['LotShape_Encoded'] = df['LotShape'].map( 
        {'Reg':2, 'IR1':1, 'IR2':1, 'IR3':1})

    # numeric encoding - LandSlope - XXX
    df['LandSlope_Encoded'] = df['LandSlope'].map( 
        {'Gtl':3, 'Mod':2, 'Sev':1})

    # numeric encoding - GarageFinish
    df['GarageFinish_Encoded'] = df['GarageFinish'].map( 
        {'Fin':3, 'RFn':2, 'Unf':1, np.NaN: 0})    
    
    # numeric encoding - GarageQual
    df['GarageQual_Encoded'] = df['GarageQual'].map( 
        {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1, np.NaN: 0})
    
    # numeric encoding - GarageQual
    df['GarageType_Encoded'] = df['GarageType'].map( 
        {'BuiltIn':4, 'Attchd':3, 'Basment':2, '2Types':2, 'Detchd': 1, 'CarPort':1, np.NaN: 0})    
    
    # numeric encoding - GarageCond XXX
    ##df['GarageCond_Encoded'] = df['GarageCond'].map( 
    ##    {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1, 'NA':np.NaN})
    
    # numeric encoding - Foundation
    df['Foundation_Encoded'] = df['Foundation'].map( 
        {'PConc':6, 'CBlock':5, 'BrkTil':4, 'Stone':3, 'Wood': 3,  'Slab':3 })
    
    # numeric encoding - Foundation
    df['MasVnrType_Encoded'] = df['MasVnrType'].map( 
        {'Stone':4, 'BrkFace':3, 'None':2, 'BrkCmn':1, np.NaN: 1 })

    # numeric encoding - Paved drive
    df['PavedDrive_Encoded'] = df['PavedDrive'].map( 
        {'Y':4, 'N':0, 'P':0, np.NaN: 1 }) 
    
    # numeric encoding - MSZoning
    df['MSZoning_Encoded'] = df['MSZoning'].map( 
        {'FV':3, 'RL':3, 'RH':2, 'RM': 2, 'c(all)': 1, np.NaN: 1 })     
    
    # numeric encoding - OverallQual
    #df['OverallQual_Encoded'] = df['OverallQual'].map( 
    #    {'0':4, '1':4, '2':4, '3':4, '4':4, '5':5, '6':6, '7':7, '8':8, '9':8, '10':8 })      

data_numeric_encoding(df_base_train)
data_numeric_encoding(df_base_test)


# %%

df_base_train['GarageArea'].unique()
df_base_train.loc[df_base_train['GarageArea'] == 0].shape
df_base_train.loc[df_base_train['GarageFinish'].isnull()].shape
df_base_train.loc[df_base_train['GarageFinish_Encoded'] == 0].shape


# %%


# %% [markdown]
# 
#  #### Numeric encoding - missing data handling

# %%
# handler for missing encoded data
    
def data_numeric_encoding_missing(df):
    
    # impute missing encoded data - LotShape_Econded
    set_missing_data_with_freq_value(df, 'GarageFinish_Encoded')    
    
    # impute missing encoded data - LotShape_Econded
    set_missing_data_with_freq_value(df, 'LotShape_Encoded')

    # impute missing encoded data - Alley_Encoded
    set_missing_data_with_freq_value(df, 'Alley_Encoded')

    # impute missing encoded data - LandSlope_Encoded xxx
    set_missing_data_with_freq_value(df, 'LandSlope_Encoded')

    # impute missing encoded data - CentralAir_Encoded
    set_missing_data_with_freq_value(df, 'CentralAir_Encoded')

    # impute missing encoded data - HeatingQC_Encoded
    set_missing_data_with_freq_value(df, 'HeatingQC_Encoded')

    # impute missing encoded data - KitchenQual_Encoded
    set_missing_data_with_freq_value(df, 'KitchenQual_Encoded')

    # impute missing encoded data - GarageQual_Encoded
    set_missing_data_with_freq_value(df, 'GarageQual_Encoded')

    # impute missing encoded data - GarageCond_Encoded - XXX
    ##set_missing_data_with_freq_value(df, 'GarageCond_Encoded')
    
    # impute missing encoded data - Foundation_Encoded
    set_missing_data_with_value(df, 'GarageType_Encoded', 0)       
    
    # impute missing encoded data - BsmtQual_Encoded
    set_missing_data_with_freq_value(df, 'BsmtQual_Encoded')

    # impute missing encoded data - GarageCond_Encoded
    ##set_missing_data_with_freq_value(df, 'BsmtCond_Encoded')

    # impute missing encoded data - BsmtExposure_Encoded
    set_missing_data_with_freq_value(df, 'BsmtExposure_Encoded')
    
    # impute missing encoded data - PoolQC_Encoded
    #set_missing_data_with_freq_value(df, 'PoolQC_Encoded')
        
    # impute missing encoded data - ExterQual_Encoded
    set_missing_data_with_freq_value(df, 'ExterQual_Encoded')

    # impute missing encoded data - ExterCond_Encoded
    set_missing_data_with_freq_value(df, 'ExterCond_Encoded')
        
    # impute missing encoded data - Foundation_Encoded
    set_missing_data_with_freq_value(df, 'Foundation_Encoded')        

    # impute missing encoded data - MSZoning
    set_missing_data_with_freq_value(df, 'MSZoning_Encoded')  
    
    # impute missing encoded data - MSZoning
    set_missing_data_with_freq_value(df, 'd_MSSubClass_Code')  
    
data_numeric_encoding_missing(df_base_train)
data_numeric_encoding_missing(df_base_test)

# %% [markdown]
#  #### Other - missing data handling

# %%
# handler for missing data

def data_missing_handler(df):

    # set MasVnrArea 0 if its null
    set_missing_data_with_value(df, 'MasVnrArea', 0)

    # set MasVnrArea 0 if its null
    set_missing_data_with_value(df, 'GarageCars', 0)    

    # set MasVnrArea 0 if its null
    set_missing_data_with_value(df, 'd_Bath', 1)
    
    
    # set MasVnrArea 0 if its null
    set_missing_data_with_value(df, 'BsmtUnfSF', 0)    
    
    

    
    
    # set LotFrontage to most frequent value if its null
    set_missing_data_with_freq_value(df, 'LotFrontage')
    
    # set TotalBsmtSF to most frequent value if its null
    set_missing_data_with_freq_value(df, 'TotalBsmtSF')
   
    # set GarageYrBlt to YearBuilt if its null
    df.loc[df['GarageYrBlt'].isnull(), ['GarageYrBlt']] = df['YearBuilt']
    
    

#data_missing_handler(df_base_train)
#data_missing_handler(df_base_test)


# %%
# exclude un-used columns

exclusions = [#'Id', 
    'd_HouseAge', 'd_RemodAge',
    'YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd', 
    'GarageArea', 'GarageYrBlt',

    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF',
    
    'Fireplaces', 'PoolArea', 'PoolQC',
    'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 
    'BsmtFinSF1', 'BsmtFinSF2',   
    '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

    'MiscFeature', 'MiscVal', 
    
    'TotRmsAbvGrd', 'GrLivArea',
    
    'MSSubClass'
    
    #'OverallCond'
] 

df_base_train = df_base_train.drop(exclusions, axis=1, errors='ignore')
df_base_test = df_base_test.drop(exclusions, axis=1, errors='ignore')


# %%



# %%
# get dataframe's metadata
df_base_train_meta = get_dataframe_metadata(df_base_train)

df_base_train_meta.head(config_df_row_correlation_count)


# %%
numeric_attributes = df_base_train_meta[np.in1d(df_base_train_meta.dtype, dtype_numeric)].index
numeric_attributes = list(set(numeric_attributes) - set(variable_ignored) - set(variable_year_month) - set(target))


# %%
# compute correlation matrix
df_base_train_corr_matrix = compute_correlation_matrix(df_base_train, target, numeric_attributes)

# plot correlations
plot_correlation_matrix(df_base_train_corr_matrix)


# %%
# get attribute's description, order by correlation
df_base_train_meta_sorted = get_df_meta_by_correlation2(
    df_base_train_corr_matrix, 
    df_base_train_meta,
    target[0])

df_base_train_meta_sorted


# %%
# plot target to features 1-d pca
plot_pca_smarter(df_base_train, df_base_train_meta_sorted, target, 15)


# %%


# %% [markdown]
#  ### Data - outliers cleaning
#  ---
# 
#  Use extreme value analysis
# 
#  Assume a Gaussian distribution and remove 3 standard deviations from the mean, for these numeric non categorical attributes
# 
#  * GrLivArea
#  * 1stFlrSF
#  * BsmtFinSF1
#  * LotArea
#  * BsmtUnfSF
# 
#  [How to identify outliers](https://machinelearningmastery.com/how-to-identify-outliers-in-your-data/)

# %%
df_clean_outlier_train = df_base_train.copy(deep=True)
df_clean_outlier_train_meta = df_base_train_meta.copy(deep=True)

df_clean_outlier_test = df_base_test.copy(deep=True)


# %%
numeric_attributes = df_clean_outlier_train_meta[np.in1d(df_clean_outlier_train_meta.dtype, dtype_numeric)].index
numeric_attributes = list(set(numeric_attributes) - set(variable_ignored) - set(variable_year_month) - set(target))


# %%
attributes = ['SalePrice', 'LotArea', 'd_TotalFlrSF', 'TotalBsmtSF'] 

for attribute in attributes:	

    upper_value = float(df_clean_outlier_train_meta.loc[df_clean_outlier_train_meta.index == attribute]['upper_3s_3'])	
    lower_value = float(df_clean_outlier_train_meta.loc[df_clean_outlier_train_meta.index == attribute]['lower_3s_3'])	
    
    df_clean_outlier_train.loc[(df_clean_outlier_train[attribute] > upper_value) | (df_clean_outlier_train[attribute] < lower_value), 'Outliers'] = attribute	
                          


# %%
df_clean_outlier_train = df_clean_outlier_train.loc[df_clean_outlier_train['Outliers'].isna()]


# %%
# get dataframe's metadata
df_clean_outlier_train_meta = get_dataframe_metadata(df_clean_outlier_train)

df_clean_outlier_train_meta.head(config_df_row_count)


# %%
attribute = 'SalePrice'

plot_attribute_chart(df_base_train, attribute)
plot_attribute_chart(df_clean_outlier_train, attribute)


# %%
attribute = 'd_TotalBldgSF'

plot_attribute_chart(df_base_train, attribute)
plot_attribute_chart(df_clean_outlier_train, attribute)


# %%
attribute = 'd_TotalFlrSF'

plot_attribute_chart(df_base_train, attribute)
plot_attribute_chart(df_clean_outlier_train, attribute)


# %%
attribute = 'LotArea'

plot_attribute_chart(df_base_train, attribute)
plot_attribute_chart(df_clean_outlier_train, attribute)


# %%
attribute = 'MasVnrArea'

plot_attribute_chart(df_base_train, attribute)
plot_attribute_chart(df_clean_outlier_train, attribute)


# %%



# %%
# compute correlation matrix
df_clean_outlier_train_corr_matrix = compute_correlation_matrix(df_clean_outlier_train, target, numeric_attributes)

# plot correlations
plot_correlation_matrix(df_clean_outlier_train_corr_matrix)


# %%
# get attribute's description, order by correlation
df_clean_outlier_train_meta_sorted = get_df_meta_by_correlation2(
    df_clean_outlier_train_corr_matrix, 
    df_clean_outlier_train_meta,
    target[0])

df_clean_outlier_train_meta_sorted


# %%
# plot target to features 1-d pca
plot_pca_smarter(df_clean_outlier_train, 
    df_clean_outlier_train_meta_sorted, 
    target, 
    15)


# %%


# %% [markdown]
#  ### Data - normalising
#  ---
# 
#  [scale-machine-learning-data](https://machinelearningmastery.com/scale-machine-learning-data-scratch-python/)
# 
#  [prepare-data-machine-learning](https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/)
# 
# 

# %%
df_clean_norm_train = df_clean_outlier_train.copy(deep=True)
df_clean_norm_train_meta = df_clean_outlier_train_meta.copy(deep=True)

df_clean_norm_test = df_clean_outlier_test.copy(deep=True)


# %%
numeric_attributes = df_clean_norm_train_meta[np.in1d(df_clean_norm_train_meta.dtype, dtype_numeric)].index
numeric_attributes = list(set(numeric_attributes) - set(variable_ignored) - set(variable_year_month) - set(target))


# %%
len(numeric_attributes)


# %%
def data_standardize(df, attribute_to_scale):

    scaler = StandardScaler()

    scaler.fit(df[attribute_to_scale])

    attributes_scaled = scaler.transform(df[attribute_to_scale])

    #df[attribute_to_scale] = attributes_scaled


data_standardize(df_clean_norm_train, numeric_attributes + target)
data_standardize(df_clean_norm_test, numeric_attributes)


# %%
#attribute_to_scale = numeric_attributes + target

#scaler = StandardScaler()

#scaler.fit(df_clean_norm_train[attribute_to_scale])

#attributes_scaled = scaler.transform(df_clean_norm_train[attribute_to_scale])

#df_clean_norm_train[attribute_to_scale] = attributes_scaled


# %%
#np.seterr(divide = 'ignore') 

#attribute_to_normalise = numeric_attributes + target

#transformer = PowerTransformer()
#transformer.fit(df_clean_norm_train[attribute_to_normalise])

#df1 = pd.DataFrame(transformer.transform(df_clean_norm_train[attribute_to_normalise]), columns=attribute_to_normalise)

#df_clean_norm_train = df_clean_norm_train.drop(attributes + target, axis=1)

#df_clean_norm_train = pd.merge(df_clean_norm_train, df1, right_index=True, left_index=True)

#df_clean_norm_train


# %%
# get dataframe's metadata
df_clean_norm_train_meta = get_dataframe_metadata(df_clean_norm_train)

df_clean_norm_train_meta.head(config_df_row_count)


# %%
attribute = 'SalePrice'

plot_attribute_chart(df_clean_outlier_train, attribute)
plot_attribute_chart(df_clean_norm_train, attribute)


# %%
#attribute = 'GrLivArea'

#plot_attribute_chart(df_clean_outlier_train, attribute)
#plot_attribute_chart(df_clean_norm_train, attribute)


# %%
attribute = 'd_TotalFlrSF'

plot_attribute_chart(df_clean_outlier_train, attribute)
plot_attribute_chart(df_clean_norm_train, attribute)


# %%
# compute correlation matrix
df_clean_norm_train_corr_matrix = compute_correlation_matrix(df_clean_norm_train, target, numeric_attributes)

# plot correlations
plot_correlation_matrix(df_clean_norm_train_corr_matrix)


# %%
# get attribute's description, order by correlation
df_clean_norm_train_meta_sorted = get_df_meta_by_correlation2(
    df_clean_norm_train_corr_matrix, 
    df_clean_norm_train_meta,
    target[0])

df_clean_norm_train_meta_sorted.head(config_df_row_correlation_count)


# %%
# plot target to features 1-d pca
plot_pca_smarter(df_clean_norm_train, df_clean_norm_train_meta_sorted, target, 19)


# %%


# %% [markdown]
#  # Feature engineering
#  ---
#  Select attributes that have the strongest relationship with the target using KBest.
# 
# 
#  [feature_selection](https://machinelearningmastery.com/feature-selection-machine-learning-python/)

# %%
# Inspect and compare f-scores (all attributes)
kbest = SelectKBest(k=len(numeric_attributes), 
    score_func=f_regression)

kbest.fit(df_clean_norm_train[numeric_attributes], 
    df_clean_norm_train[target[0]]) 

df_attribute_scores = pd.DataFrame({'attribute': numeric_attributes, 'kbest': kbest.scores_}).sort_values(by='kbest', ascending=False)
df_attribute_scores


# %%
features = df_attribute_scores.iloc[0:18, 0]

plot_pca_smarter2(df_clean_norm_train, 
    df_clean_norm_train_meta_sorted, 
    target, 
    features)


# %%



# %%


# %% [markdown]
#  # Model engineering
#  ---
# 
# 
# %% [markdown]
# ## Stochastic Gradient Descent regressor

# %%
model_number_of_features = 38
model_validation_size = 0.25
model_seed = 2

model_iteration_max = 2500
model_tollerance = 0.0005
model_alpha = 0.1

model_cv = 5


# %%
# get model features, from the top x attributes with highest k-best score
features = df_attribute_scores.iloc[0:model_number_of_features].sort_values(
    by='kbest', ascending=False)['attribute']


# %%
# allocate data for training and validation works

X = df_clean_norm_train[features] 
y = df_clean_norm_train[target]

X_train, X_validation, y_train, y_validation = train_test_split(
    X, 
    y, 
    test_size=model_validation_size, 
    random_state=model_seed)


# %%
# build pipeline

pipeline = Pipeline(
    steps = [
        ('my_scale', StandardScaler()),
        ('my_sgd', SGDRegressor(random_state=model_seed, max_iter=model_iteration_max, tol=model_tollerance, alpha=model_alpha)) 
            ])


# %%
# fit model

cv_results = cross_validate(
    pipeline, 
    X_train, 
    y_train.values.ravel(), 
    cv=model_cv, 
    verbose=False, 
    return_train_score=True, 
    return_estimator=True)

cv_results


# %%
# get model with highest validation score ('test_score')
model = cv_results['estimator'][0]


# %%
# plot learning curve

train_sizes, train_scores, val_scores = learning_curve(model,
    X_train, 
    y_train.values.ravel(), 
    cv=model_cv)

# Create means and standard deviations
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', label="Training score")
plt.plot(train_sizes, val_mean, label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# %%
# get prediction 
y_pred = model.predict(X_validation)


# %%
# compute model evaluation metrices

mse = mean_squared_error(y_validation, y_pred)
mae = mean_absolute_error(y_validation, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_validation, y_pred)

result = {
    'metric': ['MSE', 'MAE', 'RMSE', 'R2'], 
    'value': [mse, mae, rmse, r2]
}

pd.DataFrame(result)

#8.934228e-01
#8.943792e-01
#8.973478e-01

# %% [markdown]
# ### Fine tuning

# %%
# fine tune learning rate

tuning_param_grid = {
}

tuning_pipeline = Pipeline(
    steps = [
        ('my_scale', StandardScaler()),
        ('my_sgd', SGDRegressor(random_state=model_seed, max_iter=model_iteration_max, tol=model_tollerance, alpha=model_alpha)) 
            ])

tuning_cv_results = cross_validate(
    tuning_pipeline, 
    X_train, 
    y_train.values.ravel(), 
    cv=model_cv, 
    verbose=False, 
    return_train_score=True, 
    return_estimator=True)

tuning_model = tuning_cv_results['estimator'][0]

clf = GridSearchCV(tuning_model, tuning_param_grid, n_jobs=4)
clf.fit(X_train, y_train.values.ravel())

print("Best score: " + str(clf.best_score_))
print(clf.best_params_)
#pd.DataFrame(clf.cv_results_)


# %%



# %%
# plot validation data - predicited + actual result

fig, ax = plt.subplots()

size = 275 # y_validation.size

fig.set_figheight(5)
fig.set_figwidth(30)

ax.scatter(x = range(0, size), y=y_validation[0:size], c = 'blue', label = 'pred', alpha = 0.5)
ax.scatter(x = range(0, size), y=y_pred[0:size], c = 'red', label = 'act', alpha = 0.5)
plt.title('Actual and predicted values')
plt.xlabel('Observations')
plt.ylabel('price')
plt.legend()
plt.show()


# %%
# plot validation data - predicited + actual result

diff = y_validation[0:y_pred.size]['SalePrice'] - y_pred[:,]
diff.hist(bins = 40)
plt.title('Histogram of prediction errors')
plt.xlabel('Price prediction error')
plt.ylabel('Frequency')


# %%



# %%
plt.plot([0, 450000], [0, 450000], '--r')
plt.scatter(y_validation[0:size], y_pred[0:size])

plt.xlabel('Prediction', size=15)
plt.ylabel('Actual', size=15)
plt.tick_params(axis='x')
plt.tick_params(axis='y')


# %%
# model result

print(model.named_steps['my_sgd'].intercept_)

result = {
    'feature': features,
    'coef': model.named_steps['my_sgd'].coef_
}

pd.DataFrame(result)


# %%



# %%



# %%



# %%



# %%
# ... test data

X_test = df_clean_norm_test[features]
X_test.loc[:, 'Id'] = df_clean_norm_test.loc[:, 'Id']

#X_test.reset_index(drop=True, inplace=True)
X_test.set_index("Id", drop=True, inplace=True)


# %%
# get prediction - use model with highest validation score ('test_score')

y_pred = model.predict(X_test)

res = X_test
res['SalePrice'] = y_pred

# create submission file
z = res.reset_index()
zz = z[['Id', 'SalePrice']].sort_values(by='Id', ascending=True, na_position='first').reset_index()
zz[['Id', 'SalePrice']].to_csv('submission.csv',index=False)


# %%



# %%
now = datetime.now()

with pd.ExcelWriter('output_' + now.strftime("%d%m%Y_%H%M%S") + '.xlsx') as writer:
    df_raw_train.to_excel(writer, sheet_name='train')
    df_raw_train_meta_sorted.to_excel(writer, sheet_name='train_corr')
    df_base_train.to_excel(writer, sheet_name='train_base')
    df_base_train_meta_sorted.to_excel(writer, sheet_name='train_base_corr')
    df_clean_outlier_train.to_excel(writer, sheet_name='train_clean')
    df_clean_outlier_train_meta_sorted.to_excel(writer, sheet_name='train_clean_corr')


# %%



# %%



# %%



# %%



# %%



# %%


