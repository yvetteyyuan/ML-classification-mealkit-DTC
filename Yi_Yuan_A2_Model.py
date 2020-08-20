#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing timeit
import timeit

code_to_test="""
# timeit

# Student Name : Yi Yuan
# Cohort       : 4

################################################################################
# Import Packages
################################################################################
 
import numpy as np                                   # highlevel math functions
import random            as rand                     # random number gen
import pandas            as pd                       # data science essentials
import matplotlib.pyplot as plt                      # data visualization
import seaborn           as sns                      # enhanced data viz
from sklearn.model_selection import train_test_split # train-test split
import statsmodels.formula.api as smf                # regression modeling
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.metrics import confusion_matrix         # confusion matrix
from sklearn.metrics import roc_auc_score            # auc score
from sklearn.neighbors import KNeighborsClassifier   # KNN for classification
from sklearn.neighbors import KNeighborsRegressor    # KNN for regression
from sklearn.preprocessing import StandardScaler     # standard scaler
from sklearn.tree import DecisionTreeClassifier      # classification trees
from sklearn.tree import export_graphviz             # exports graphics
from sklearn.externals.six import StringIO           # saves objects in memory
from IPython.display import Image                    # displays on frontend
import pydotplus                                     # interprets dot objects
from sklearn.model_selection import GridSearchCV     # hyperparameter tuning
from sklearn.metrics import make_scorer              # customizable scorer
from sklearn.ensemble import RandomForestClassifier     # random forest
from sklearn.ensemble import GradientBoostingClassifier # gbm

###############################################################################
# Load Data
################################################################################

original_df = pd.read_excel('Apprentice_Chef_Dataset.xlsx')

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

########################
# create family groups #

# impute missing values
original_df['FAMILY_NAME'] = original_df['FAMILY_NAME'].fillna('NA')

# check to see result
# print(original_df.isnull().sum().sum())

# check the top largest families
#lst = original_df['FAMILY_NAME'].value_counts()

# placeholder list
placeholder_lst = []

# looping to family groups
for name in original_df.loc[:, 'FAMILY_NAME']:
    if 'Tully' in name:
        placeholder_lst.append('Tully')
    elif 'Frey' in name:
        placeholder_lst.append('Tully')
    elif 'Arryn' in name:
        placeholder_lst.append('Tully')
    elif 'Targaryen' in name:
        placeholder_lst.append('Targaryen')
    elif 'Lannister' in name:
        placeholder_lst.append('Lannister')
    elif 'Baratheon' in name:
        placeholder_lst.append('Baratheon')
    elif 'Tyrell' in name:
        placeholder_lst.append('Tyrell')
    elif 'Stark' in name:
        placeholder_lst.append('Stark')
    elif 'Snow' in name:
        placeholder_lst.append('Stark') 
    elif 'Greyjoy' in name:
        placeholder_lst.append('Greyjoy') 
    elif 'Martell' in name:
        placeholder_lst.append('Martell')
    elif 'Sand' in name:
        placeholder_lst.append('Martell')
    else: 
        placeholder_lst.append('Other')

# converting list into a series
original_df['FAMILY_GROUP'] = pd.Series(placeholder_lst)

# checking results
# original_df.loc[:,'FAMILY_GROUP'].value_counts()

#########################
# split personal emails #

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in original_df.iterrows():
    
    # splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    
# converte placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)

# displaying the results
# email_df

#######################################
# concatenate with original DataFrame #

# renaming column to concatenate
email_df.columns = ['name' , 'EMAIL_DOMAIN']

# concatenating personal_email_domain with friends DataFrame
original_df = pd.concat([original_df, email_df.loc[:, 'EMAIL_DOMAIN']],
                    axis = 1)  # 0 for row, 1 for col

# printing value counts of personal_email_domain
original_df.loc[: ,'EMAIL_DOMAIN'].value_counts()

##################################################
# aggregate domains into higher-level categories #

# email domain types
personal_email_domains = ['gmail.com', 'protonmail.com', 'yahoo.com',
                          'msn.com','aol.com', 'passport.com',
                          'hotmail.com', 'live.com', 'me.com']
corporate_email_domains  = ['amex.com', 'cocacola.com', 'merck.com',
                          'mcdonalds.com', 'jnj.com', 'nike.com', 'apple.com', 
                          'dupont.com', 'ge.org', 'ibm.com', 'chevron.com', 
                          'microsoft.com', 'unitedhealth.com','exxon.com', 
                          'travelers.com','boeing.com', 'caterpillar.com', 
                          'pg.com','verizon.com', 'mmm.com', 'disney.com', 
                          'walmart.com', 'pfizer.com', 'visa.com', 
                          'jpmorgan.com', 'goldmansacs.com', 'cisco.com',
                          'unitedtech.com', 'intel.com', 'homedepot.com']

# placeholder list
placeholder_lst = []

# looping to group observations by domain type
for domain in original_df['EMAIL_DOMAIN']:
        if domain in personal_email_domains:
            placeholder_lst.append('personal')
            
        elif domain in corporate_email_domains:
            placeholder_lst.append('corporate')
            
        else:
            print('Unknown')

# concatenating with original DataFrame
original_df['DOMAIN_GROUP'] = pd.Series(placeholder_lst)

# checking results
# original_df['DOMAIN_GROUP'].value_counts()

#################################
### part 1 - outlier analysis ###
#################################

# setting outlier thresholds
REVENUE_hi      = 2400
TOTAL_MEALS_ORDERED_lo = 15
TOTAL_MEALS_ORDERED_hi  = 120
CONTACTS_W_CUSTOMER_SERVICE_hi  = 10
AVG_TIME_PER_SITE_VISIT_hi  = 180
WEEKLY_PLAN_lo  = 1
WEEKLY_PLAN_hi = 14
CANCELLATIONS_BEFORE_NOON_hi  = 5
EARLY_DELIVERIES_lo = 1
EARLY_DELIVERIES_hi = 5
LATE_DELIVERIES_lo = 10
AVG_PREP_VID_TIME_lo    = 60
AVG_PREP_VID_TIME_hi    = 190
LARGEST_ORDER_SIZE_lo     = 1
LARGEST_ORDER_SIZE_hi     = 9
MASTER_CLASSES_ATTENDED_hi     = 1
MEDIAN_MEAL_RATING_lo = 2
MEDIAN_MEAL_RATING_hi  = 4
AVG_CLICKS_PER_VISIT_lo  = 7.5
AVG_CLICKS_PER_VISIT_hi  = 20
TOTAL_PHOTOS_VIEWED_lo = 1

##############################################################################
## Feature Engineering (outlier thresholds)                                 ##
##############################################################################

# developing features (columns) for outliers

# Revenue
original_df['out_REVENUE'] = 0
condition_hi = original_df.loc[0:,'out_REVENUE'][original_df['REVENUE'] > REVENUE_hi]

original_df['out_REVENUE'].replace(to_replace = condition_hi,
                            value      = 1,
                            inplace    = True)

# Total meals ordered
original_df['out_TOTAL_MEALS_ORDERED'] = 0
condition_lo = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_lo]
condition_hi = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]

original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_lo,
                                         value      = 1,
                                         inplace    = True)
original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                        value      = 1,
                                        inplace    = True)

# Contacts with customer service
original_df['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]

original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                                value      = 1,
                                                inplace    = True)

# Average time per site visit
original_df['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]

original_df['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                            value      = 1,
                                            inplace    = True)

# Weekly plan
original_df['out_WEEKLY_PLAN'] = 0
condition_lo = original_df.loc[0:,'out_WEEKLY_PLAN'][original_df['WEEKLY_PLAN'] < WEEKLY_PLAN_lo]
condition_hi = original_df.loc[0:,'out_WEEKLY_PLAN'][original_df['WEEKLY_PLAN'] > WEEKLY_PLAN_hi]

original_df['out_WEEKLY_PLAN'].replace(to_replace = condition_lo,
                                         value      = 1,
                                         inplace    = True)
original_df['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# Cancellations before noon
original_df['out_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = original_df.loc[0:,'out_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_hi]

original_df['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                                value      = 1,
                                                inplace    = True)

# Early deliveries
original_df['out_EARLY_DELIVERIES'] = 0
condition_lo = original_df.loc[0:,'out_EARLY_DELIVERIES'][original_df['EARLY_DELIVERIES'] < EARLY_DELIVERIES_lo]
condition_hi = original_df.loc[0:,'out_EARLY_DELIVERIES'][original_df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]

original_df['out_EARLY_DELIVERIES'].replace(to_replace = condition_lo,
                                         value      = 1,
                                         inplace    = True)
original_df['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# Late deliveries
original_df['out_LATE_DELIVERIES'] = 0
condition_lo = original_df.loc[0:,'out_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] < LATE_DELIVERIES_lo]

original_df['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)


# Average prep video time
original_df['out_AVG_PREP_VID_TIME'] = 0
condition_lo = original_df.loc[0:,'out_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] < AVG_PREP_VID_TIME_lo]
condition_hi = original_df.loc[0:,'out_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]

original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_lo,
                                     value      = 1,
                                     inplace    = True)
original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                     value      = 1,
                                     inplace    = True)

# Largest order size
original_df['out_LARGEST_ORDER_SIZE'] = 0
condition_lo = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_lo]
condition_hi = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo,
                                         value      = 1,
                                         inplace    = True)
original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                         value      = 1,
                                         inplace    = True)

# Median meal rating
original_df['out_MEDIAN_MEAL_RATING'] = 0
condition_lo = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] < MEDIAN_MEAL_RATING_lo]
condition_hi = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_hi]

original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_lo,
                                         value      = 1,
                                         inplace    = True)
original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,
                                         value      = 1,
                                         inplace    = True)

# Average clicks per visit
original_df['out_AVG_CLICKS_PER_VISIT'] = 0
condition_lo = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]
condition_hi = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                         value      = 1,
                                         inplace    = True)
original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                         value      = 1,
                                         inplace    = True)

# Master classes attended
original_df['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = original_df.loc[0:,'out_MASTER_CLASSES_ATTENDED'][original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_hi]

original_df['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                             value      = 1,
                                             inplace    = True)

# Total photos viewed
original_df['out_TOTAL_PHOTOS_VIEWED'] = 0
condition_lo = original_df.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] < TOTAL_PHOTOS_VIEWED_lo]

original_df['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_lo,
                                             value      = 1,
                                             inplace    = True)
#####################################
### part 2 - trend based features ###
#####################################

# setting trend-based thresholds

TOTAL_MEALS_ORDERED_change_hi  = 15 # trend changes above this point
UNIQUE_MEALS_PURCH_change_hi = 9 # data scatters above this point
CONTACTS_W_CUSTOMER_SERVICE_change_hi  = 10 # trend changes above this point
AVG_TIME_PER_SITE_VISIT_change_hi  = 300 # data scatters above this point
CANCELLATIONS_BEFORE_NOON_change_hi  = 5 # data scatters above this point
CANCELLATIONS_AFTER_NOON_change_hi  = 2 # trend changes above this point
MOBILE_LOGINS_change_hi = 6 # trend changes above this point
PC_LOGINS_change_hi = 2 # trend changes above this point
LATE_DELIVERIES_change_hi = 10 # data scatters above this point
AVG_PREP_VID_TIME_change_hi = 300 # data scatters above this point
LARGEST_ORDER_SIZE_change_hi = 8 # trend changes above this point
MASTER_CLASSES_ATTENDED_change_hi  = 2 # trend changes above this point
MEDIAN_MEAL_RATING_change_hi = 4 # trend changes above this point

MOBILE_LOGINS_change_lo = 4 # data scatters below this point
AVG_CLICKS_PER_VISIT_change_lo  = 10 # data scatters below this point
PC_LOGINS_change_lo = 1 # data scatters below this point

TOTAL_PHOTOS_VIEWED_change_at   = 0 # zero inflated

# developing features (columns) for trend based thresholds

########################################
## change above threshold             ##
########################################

# greater than sign
 
# Total meals ordered
original_df['change_TOTAL_MEALS_ORDERED'] = 0
condition = original_df.loc[0:,'change_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_change_hi]

original_df['change_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                            value      = 1,
                                            inplace    = True)
# Unique meals purchased
original_df['change_UNIQUE_MEALS_PURCH'] = 0
condition = original_df.loc[0:,'change_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_change_hi]

original_df['change_UNIQUE_MEALS_PURCH'].replace(to_replace = condition,
                                            value      = 1,
                                            inplace    = True)

# Contacts with customer service
original_df['change_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition = original_df.loc[0:,'change_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_change_hi]

original_df['change_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition,
                                                    value      = 1,
                                                    inplace    = True)

# Average time per site visit
original_df['change_AVG_TIME_PER_SITE_VISIT'] = 0
condition = original_df.loc[0:,'change_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_change_hi]

original_df['change_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)

# Cancellations before noon
original_df['change_CANCELLATIONS_BEFORE_NOON'] = 0
condition = original_df.loc[0:,'change_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_change_hi]

original_df['change_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)

# Cancellations after noon
original_df['change_CANCELLATIONS_AFTER_NOON'] = 0
condition = original_df.loc[0:,'change_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_change_hi]

original_df['change_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)

# Mobile logins
original_df['change_MOBILE_LOGINS'] = 0
condition = original_df.loc[0:,'change_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] > MOBILE_LOGINS_change_hi]

original_df['change_MOBILE_LOGINS'].replace(to_replace = condition,
                                        value      = 1,
                                        inplace    = True)

# PC logins
original_df['change_PC_LOGINS'] = 0
condition = original_df.loc[0:,'change_PC_LOGINS'][original_df['PC_LOGINS'] > PC_LOGINS_change_hi]

original_df['change_PC_LOGINS'].replace(to_replace = condition,
                                    value      = 1,
                                    inplace    = True)

# Late deliveries
original_df['change_LATE_DELIVERIES'] = 0
condition = original_df.loc[0:,'change_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_change_hi]

original_df['change_LATE_DELIVERIES'].replace(to_replace = condition,
                                        value      = 1,
                                        inplace    = True)

# Average prep video time
original_df['change_AVG_PREP_VID_TIME'] = 0
condition = original_df.loc[0:,'change_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_change_hi]

original_df['change_AVG_PREP_VID_TIME'].replace(to_replace = condition,
                                        value      = 1,
                                        inplace    = True)

# Largest order size
original_df['change_LARGEST_ORDER_SIZE'] = 0
condition = original_df.loc[0:,'change_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_change_hi]

original_df['change_LARGEST_ORDER_SIZE'].replace(to_replace = condition,
                                            value      = 1,
                                            inplace    = True)

# Median meal rating
original_df['change_MEDIAN_MEAL_RATING'] = 0
condition = original_df.loc[0:,'change_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_change_hi]

original_df['change_MEDIAN_MEAL_RATING'].replace(to_replace = condition,
                                            value      = 1,
                                            inplace    = True)

########################################
## change at threshold                ##
########################################

# double-equals sign

# Total photos viewed
original_df['change_TOTAL_PHOTOS_VIEWED'] = 0
condition = original_df.loc[0:,'change_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] == TOTAL_PHOTOS_VIEWED_change_at]

original_df['change_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                            value      = 1,
                                            inplace    = True)

# checking results
# original_df.head(n = 10)

########################################
## change below threshold             ##
########################################

# less than sign

# Mobile logins
condition = original_df.loc[0:,'change_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] < MOBILE_LOGINS_change_lo]

original_df['change_MOBILE_LOGINS'].replace(to_replace = condition,
                                        value      = 1,
                                        inplace    = True)

# PC logins
condition = original_df.loc[0:,'change_PC_LOGINS'][original_df['PC_LOGINS'] < PC_LOGINS_change_lo]

original_df['change_PC_LOGINS'].replace(to_replace = condition,
                                    value      = 1,
                                    inplace    = True)

# Average clicks per visit
original_df['change_AVG_CLICKS_PER_VISIT'] = 0
condition = original_df.loc[0:,'change_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_change_lo]

original_df['change_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition,
                                            value      = 1,
                                            inplace    = True)
#################################
### part 3 - categorical data ###
#################################

# one hot encoding categorical variables
one_hot_FAMILY_GROUP       = pd.get_dummies(original_df['FAMILY_GROUP'])
one_hot_EMAIL_DOMAIN       = pd.get_dummies(original_df['EMAIL_DOMAIN'])
one_hot_DOMAIN_GROUP       = pd.get_dummies(original_df['DOMAIN_GROUP'])

# dropping categorical variables after they've been encoded
original_df = original_df.drop('FAMILY_GROUP', axis = 1)
original_df = original_df.drop('EMAIL_DOMAIN', axis = 1)
original_df = original_df.drop('DOMAIN_GROUP', axis = 1)

# joining codings together
original_df = original_df.join([one_hot_FAMILY_GROUP, one_hot_EMAIL_DOMAIN, one_hot_DOMAIN_GROUP])

# saving new columns
new_columns = original_df.columns

# dropping categorical variables 
original_df = original_df.drop(labels = ['NAME','EMAIL','FIRST_NAME', 'FAMILY_NAME'], 
                     axis = 1)

################################################################################
# Train/Test Split
################################################################################

# declaring explanatory variables
apchef_explanatory = original_df.drop(labels='CROSS_SELL_SUCCESS', 
                            axis=1)

# declaring response variable
apchef_target = original_df.loc[:,'CROSS_SELL_SUCCESS']

# updating explanatory variables in scikit-learn 
x_variables = ['AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER', 
               'CANCELLATIONS_AFTER_NOON', 'TASTES_AND_PREFERENCES',
               'PC_LOGINS', 'REFRIGERATED_LOCKER', 
               'FOLLOWED_RECOMMENDATIONS_PCT', 'change_LATE_DELIVERIES',
               'Greyjoy', 'jpmorgan.com', 'microsoft.com', 'msn.com']

# updating variable DataFrame based the the analysis above
apchef_explanatory  = apchef_explanatory.loc[ : , x_variables]

# train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
            apchef_explanatory,
            apchef_target,
            test_size = 0.25,
            random_state = 219, 
            stratify = apchef_target)

# merging training data for statsmodels
apchef_train = pd.concat([X_train, y_train], axis = 1)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# INSTANTIATING the model object without hyperparameters
gbm_default = GradientBoostingClassifier(loss          = 'deviance',
                                              learning_rate = 0.1,
                                              n_estimators  = 100,
                                              criterion     = 'friedman_mse',
                                              max_depth     = 3,
                                              warm_start    = False,
                                              random_state  = 802)

# FIT step is needed as we are not using .best_estimator
gbm_default_fit = gbm_default.fit(X_train, y_train)

# PREDICTING based on the testing set
gbm_default_pred = gbm_default_fit.predict(X_test)

################################################################################
# Final Model Score (score)
################################################################################

test_score = roc_auc_score(y_true  = y_test, y_score = gbm_default_pred).round(4)

print('AUC Score:',  roc_auc_score(y_true  = y_test, y_score = gbm_default_pred).round(4))

test_score = 0.7338

"""

# calculating execution time
elapsed_time=timeit.timeit(code_to_test, number =3)/3
print(elapsed_time)


# In[ ]:




