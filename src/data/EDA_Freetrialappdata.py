

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

print("number of unique apps in free trial app data", free_Trial_app_data['app_name'].nunique())   # 27236

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
print("number of apps with size 1", appnamebytrialdate_size1['app_name'].nunique())
appnamebytrialdate_size1_new = free_Trial_app_data[free_Trial_app_data['app_name'].isin(appnamebytrialdate_size1['app_name'])].sort_values(by=['app_name','trial_date'])

print("number of apps with size 1", appnamebytrialdate_size1_new['app_name'].nunique())

#free trial appdata where app occurance is twice

# appnamebytrialdate_size2_new['missingvalues_rowwise']=appnamebytrialdate_size2_new.isnull().sum(axis=1)
#
# print(appnamebytrialdate_size2_new.head())
#
# cleaneddf=pd.DataFrame(columns=appnamebytrialdate_size2_new.columns)
# for each in appnamebytrialdate_size2_new:
#     if each['app_name'] not in cleaneddf['app_name']:
#         cleaneddf.append(each)
#     else:
#         oldvalues = cleaneddf[each['app_name'],'missingvalues_rowwise']
#         newvalues = each['missingvalues_rowwise']
#         if(oldvalues[1]>newvalues):
#             appnamebytrialdate_size2_new.drop()





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


#accessing usage data

from google.cloud import bigquery
from datetime import date
client = bigquery.Client()


#print(goodapps['trial_date'].dt.date,goodapps['trial_end_date'].dtype)

#sqlbigquery_data = "SELECT u.* FROM `infusionsoft-looker-poc.asu_msba_free_trial_conversion.CONFIDENTIAL_usage_data` u where u.appname = 'ac505' and u.date BETWEEN CAST( " + goodapps['trial_date'] + "AS DATE) and CAST( "+ goodapps['trial_end_date'] + "AS DATE)"

sqlbigquery_data = """SELECT u.* FROM `infusionsoft-looker-poc.asu_msba_free_trial_conversion.CONFIDENTIAL_usage_data` u
join `infusionsoft-looker-poc.asu_msba_free_trial_conversion.CONFIDENTIAL_free_trail_apps_table`  a
on u.appname = a.app_name 
where u.appname not in ('tl471','wd410','mw416','kw563','ov450') and u.date BETWEEN DATE (a.trial_date, "America/Los_Angeles") and 
IF(date_diff(Date(new_customer_date, "America/Los_Angeles"), Date(trial_date, "America/Los_Angeles"),day)<=13,date(a.new_customer_date,"America/Los_Angeles") ,DATE_ADD( date(a.trial_date,"America/Los_Angeles"), interval 13 DAY))"""


usagedatasetfree= pd.read_gbq(sqlbigquery_data,
                project_id='infusionsoft-looker-poc',
                dialect='standard'
                )
#
# read= client.query(query,location='US',job_config=job_config)
#UsageData = usagedatasetfree.result().to_dataframe()


print(len(usagedatasetfree))

print(usagedatasetfree['appname'].nunique())

listofgoodappnames = goodapps['app_name'].unique()

print(len(listofgoodappnames))

usagedatasetfree1 = usagedatasetfree[usagedatasetfree.appname.isin(listofgoodappnames)]


print(len(usagedatasetfree1))


print(usagedatasetfree1['appname'].nunique())

print(usagedatasetfree1.isna().sum()/usagedatasetfree1.shape[0])

usagedatasetfree1.drop(['NUMEMAILSSENT_BROADCAST','NUMCONTACTSSENT_BROADCAST','NUMEMAILSSENT_NULL','NUMCONTACTSSENT_NULL','NUMEMAILSOPENED_BROADCAST','NUMCONTACTSOPENED_BROADCAST'],axis=1,inplace=True)
usagedatasetfree1.drop(['NUMEMAILSCLICKED_BROADCAST','NUMCONTACTSCLICKED_BROADCAST','NUMEMAILSCLICKED_AUTO_SYSTEM','NUMCONTACTSCLICKED_AUTO_SYSTEM'],axis=1,inplace=True)



usagedatasetfree2 = usagedatasetfree1.sort_values(by=['appname','date'])

print(len(usagedatasetfree2))

usagedatasetfree2.drop_duplicates(keep=False,inplace=True)

print(usagedatasetfree2['appname'].nunique())

#samplegoodapps



usagedatasetfree2_counts = usagedatasetfree2.groupby(['appname']).size().to_frame('size').reset_index().sort_values(['size'], ascending=[True])




# below code is for taking a sample
# mergedata = pd.merge(usagedatasetfree2_counts,goodapps,how='left',left_on='appname',right_on='app_name')
#
# mergedata =mergedata[['appname','size','Target']]
#
#
# cc469=usagedatasetfree2[usagedatasetfree2['appname'].isin(['cc469','od612','fh456','zv446','fe606','sc497','zv598','ac506','ix616','th404','lj402','ik402'])]
#
# print(len(mergedata[(mergedata['size']==8) & (mergedata['Target']==1)])/len(mergedata[mergedata['size']==8]))
#
#
#
#
#
# # take random sample from free trial data from usagedata
#
# def stratified_sample_df(df, col, n_samples):
#     #n = min(n_samples, df[col].value_counts().max())
#     df_ = df.groupby(col).apply(lambda x: x.sample(min(len(x),n_samples)))
#     df_.index = df_.index.droplevel(0)
#     return df_
#
# sample = stratified_sample_df(mergedata, 'size', 200)
#
#
# sample_appnameslist = list(sample['appname'])
#
# print(sample['Target'].value_counts())
#
#
#
# sampleusagedata = usagedatasetfree2[usagedatasetfree2.appname.isin(sample_appnameslist)]
#
# print(sampleusagedata.info())

#impute missing values with zero for sampleusagedata

print(len(usagedatasetfree2))

from sklearn.impute import SimpleImputer

si=SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

usagedataimp = pd.DataFrame(si.fit_transform(usagedatasetfree2),index=usagedatasetfree2.index,columns=usagedatasetfree2.columns)

print(usagedataimp.info())

print(usagedataimp.count())


print(usagedataimp.isna().sum()/usagedataimp.shape[0])

#usagedataimp.drop(['NUMEMAILSSENT_BROADCAST','NUMCONTACTSSENT_BROADCAST','NUMEMAILSSENT_NULL','NUMCONTACTSSENT_NULL'])


usagedataimpnum = usagedataimp.drop(['date'],axis=1)

usagedataimpnum.drop(['NUMCONTACTSOPENED_NULL','NUMCONTACTSCLICKED_NULL'],axis=1,inplace=True)

print(usagedataimpnum['appname'].nunique())



usagedataimpnum.to_csv(r"/Users/shravanamee/Desktop/Free-Trial-Conversion-New/infusionsoft-free-trial/src/data/usagedataimpall.csv")

usagedataimpnumunique = usagedataimpnum.groupby(['appname','free_email']).apply(lambda x:x.sum()/len(x)).reset_index()

nulllist=[]
for col in usagedataimpnumunique.columns:
    if usagedataimpnumunique[col].sum()==0:
        nulllist.append(col)

print(len(usagedataimpnumunique))


usagedataimpnumunique.drop(['NUMCONTACTSCLICKED_NULL', 'NUMCONTACTSOPENED_NULL'],axis=1,inplace=True)



usagedataimpnumunique.to_csv(r"/Users/shravanamee/Desktop/Free-Trial-Conversion-New/infusionsoft-free-trial/src/data/usagedataimpnumunique.csv")



# getting sample data for same apps from goodapps

goodappssample = goodapps[goodapps.app_name.isin(sample_appnameslist)]

print(goodappssample.info())

goodappssample.drop(['account_id','opportunity_id','contact_id','opportunity_owner_name'],axis=1,inplace=True)

#will consider appname later

catfreatures= ['opportunity_stage_name','is_free_trial_initiated','contact_lead_source','Target',
               'contact_phase','lead_lead_source','initial_edition','current_edition_type','kickstart_owner_name','kickstart_owner_role','kickstart_type','cancel_eligible','contract_status','edition_category','promo_customer']

datefeatures = ['opportunity_demo_date','trial_date','lead_converted_date','new_customer_date','start_date','sales_cohort_date','paying_customer_date',
                'lost_revenue_date','in_trial_first_email_date','in_trial_last_email_date','first_login_on','second_login','first_contact_on','last_contact_on',
                'in_trial_first_campaign_published_on','in_trial_last_campaign_published_on','in_trial_first_campaign_launched_on','in_trial_last_campaign_launched_on',
                'trial_end_date']

nufeatures = ['count_of_active_users_in_last_6_months','initial_mrr_post_promo','current_mrr','initial_mrr','in_trial_total_emails_sent',
              'in_trial_count_of_logins','in_trial_count_of_contacts_added','in_trial_count_of_published_campaigns','in_trial_count_of_launched_campaigns']


for c in catfreatures:
    print("----------")
    print(c,"----",goodapps[c].nunique())


goodapps.to_csv(r"/Users/shravanamee/Desktop/Free-Trial-Conversion-New/infusionsoft-free-trial/src/data/goodapps.csv")
print(goodapps[goodapps['opportunity_stage_name']=="Closed Won"][['app_name','Target','trial_date','trial_end_date']])



import seaborn as sns


#sns.stripplot(x="contact_lead_source", y="Target", data=goodapps)

#sns.pairplot(goodappssample[catfreatures])









# merging both the sample data

finalsampledata =  pd.merge(sampleimpunique,goodapps,how='left',left_on='appname',right_on='app_name')

finalsampledata = pd.merge(finalsampledata,mergedata,how='left',left_on='appname',right_on='appname')


finalsampledata.drop(['app_name','Target_y'],axis=1,inplace=True)

print(finalsampledata['initial_edition'].value_counts())
print(finalsampledata['current_edition_type'].value_counts())
print(goodapps['kickstart_type'].value_counts())



