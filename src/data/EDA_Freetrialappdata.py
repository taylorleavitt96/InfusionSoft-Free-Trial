

import numpy as np
from google.cloud import bigquery
import google.auth
import pandas as pd


free_Trial_app_data = pd.read_csv(r'/Users/shravanamee/Desktop/Free-Trial-Conversion-New/infusionsoft-free-trial/src/data/CONFIDENTIAL_free_trial_apps_000000000000.csv')

head=free_Trial_app_data.head()

print(free_Trial_app_data.describe())

# deleting test appnames
free_Trial_app_data = free_Trial_app_data[~free_Trial_app_data['app_name'].isin(['tl471','wd410', 'mw416', 'kw563', 'ov450'])]


# dropping duplicates

free_Trial_app_data.drop_duplicates(keep=False,inplace=True)

print("free trial appdata describe ", free_Trial_app_data.describe())

print("number of unique apps in free trial app data", free_Trial_app_data['app_name'].nunique())   # 27242

# grouping by appname and trialdate

appnamebytrialdate = free_Trial_app_data.groupby(['app_name','trial_date']).size().to_frame('size').reset_index().sort_values(['size'], ascending=[False])

print("number of apps ", appnamebytrialdate['app_name'].nunique())

# getting the list of appnames that have more than one occurance on a day

appnamebytrialdate_size2 = appnamebytrialdate[appnamebytrialdate['size'] == 2]
print("number of apps with size 2", appnamebytrialdate_size2['app_name'].nunique())
appnamebytrialdate_size2_new = free_Trial_app_data[free_Trial_app_data['app_name'].isin(appnamebytrialdate_size2['app_name'])].sort_values(by=['app_name','trial_date'])

appnamebytrialdate_size3 = appnamebytrialdate[appnamebytrialdate['size'] == 3]
print("number of apps with size 3", appnamebytrialdate_size3['app_name'].nunique())
appnamebytrialdate_size3_new = free_Trial_app_data[free_Trial_app_data['app_name'].isin(appnamebytrialdate_size3['app_name'])].sort_values(by=['app_name','trial_date'])

appnamebytrialdate_size4 = appnamebytrialdate[appnamebytrialdate['size'] == 4]
print("number of apps with size 4", appnamebytrialdate_size4['app_name'].nunique())
appnamebytrialdate_size4_new = free_Trial_app_data[free_Trial_app_data['app_name'].isin(appnamebytrialdate_size4['app_name'])].sort_values(by=['app_name','trial_date'])

appnamebytrialdate_size6 = appnamebytrialdate[appnamebytrialdate['size'] == 6]
print("number of apps with size 6", appnamebytrialdate_size6['app_name'].nunique())
appnamebytrialdate_size6_new = free_Trial_app_data[free_Trial_app_data['app_name'].isin(appnamebytrialdate_size6['app_name'])].sort_values(by=['app_name','trial_date'])

appnamebytrialdate_size8 = appnamebytrialdate[appnamebytrialdate['size'] == 8]
print("number of apps with size 8", appnamebytrialdate_size8['app_name'].nunique())
appnamebytrialdate_size8_new = free_Trial_app_data[free_Trial_app_data['app_name'].isin(appnamebytrialdate_size8['app_name'])].sort_values(by=['app_name','trial_date'])


print("number of apps with more than one occurance", appnamebytrialdate['app_name'].nunique())

# get free trial info for those with more than one occurrance

# appnamebytrialdatenew = free_Trial_app_data[free_Trial_app_data['app_name'].isin(appnamebytrialdate['app_name'])].sort_values(by=['app_name','trial_date'])
#
# appnamebytrialdatenew2 = appnamebytrialdatenew.groupby(['app_name','trial_date']).size().to_frame('size').reset_index().sort_values(['size'], ascending=[False])
#
# print("appnamebytrialdatenew describe", appnamebytrialdatenew2)




#free trial appdata where app occurance is one per day


appnamebytrialdate_size1 = appnamebytrialdate[appnamebytrialdate['size'] == 1]
print("number of apps with size 8", appnamebytrialdate_size1['app_name'].nunique())
appnamebytrialdate_size1_new = free_Trial_app_data[free_Trial_app_data['app_name'].isin(appnamebytrialdate_size1['app_name'])].sort_values(by=['app_name','trial_date'])


# apps those have trial date after new customer date

maliciousapps = appnamebytrialdate_size1_new[appnamebytrialdate_size1_new['new_customer_date'] < appnamebytrialdate_size1_new['trial_date']]

print("number of apps those have trial date after new customer dates", maliciousapps['app_name'].nunique())


# apps that have single occurance and trial date is before new customer date

goodapps = appnamebytrialdate_size1_new[~appnamebytrialdate_size1_new['app_name'].isin(maliciousapps['app_name'])]

# number of good apps

print("number of good apps", goodapps['app_name'].nunique())


# setting target column
goodapps['Target'] = list(map((lambda x: 0 if pd.isna(x) else 1),goodapps['new_customer_date']))

print(goodapps.info())
print(goodapps['Target'].value_counts())  # 0-23466 1-276


# converting trial_date and new_customer_date  type objects to datetime objects

import datetime

goodapps['trial_date'] = pd.to_datetime(goodapps['trial_date'])
goodapps['new_customer_date'] = pd.to_datetime(goodapps['new_customer_date'])

print(goodapps.info())


# calculating trial_end_date column

datediff = 0

def datediff(trialdate,newcustomerdate):

    if pd.isna(newcustomerdate):

        return trialdate+datetime.timedelta(13,0,0)

    else:
        datediff=(newcustomerdate-trialdate).days
        if datediff < 14:

            return newcustomerdate
        else:

            return trialdate+datetime.timedelta(13,0,0)


goodapps['trial_end_date'] = goodapps.apply(lambda row: datediff(row['trial_date'], row['new_customer_date']), axis=1)

print(goodapps.info())
print(goodapps['app_name'].nunique())
# calcualting missing values %

print(goodapps.isna().sum()/goodapps.shape[0])

print(type(goodapps['trial_end_date']),type(goodapps['trial_date']))


# checking day difference
#
# goodapps['days'] = goodapps['trial_end_date']-goodapps['trial_date']
# goodapps['days']= goodapps['days']/np.timedelta64(1,'D')
# print(goodapps[goodapps['days'] > 13])
# goodapps.drop(['days'], axis=1)


# drop mrr columns

goodapps.drop(['initial_mrr_post_promo','current_mrr','initial_mrr'], axis=1)


# for each in goodapps.columns.values:
#     print(each, goodapps[each].nunique())
#
# y = goodapps['Target']
# X = goodapps.drop(['Target'], axis=1)
#
#
# from sklearn.model_selection import train_test_split
#
# Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, test_size=0.2,random_state=1)
#
# Xtrain = Xtrain.copy()
# Xtest = Xtest.copy()
# ytrain = ytrain.copy()
# ytest = ytest.copy()


# accessing usage data

from google.cloud import bigquery

client = bigquery.Client()

print(goodapps['trial_date'].date(),goodapps['trial_end_date'].date())

sqlbigquery_data = """SELECT u.* FROM asu_msba_free_trial_conversion.CONFIDENTIAL_usage_data u where u.appname='ac505' and  u.date BETWEEN %s and %s """ % (goodapps['trial_date'],goodapps['trial_end_date'])

usagedatasetfree= pd.read_gbq(sqlbigquery_data,
                project_id='infusionsoft-looker-poc',
                dialect='standard'
                )
