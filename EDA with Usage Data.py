
# coding: utf-8

# In[3]:


import pandas as pd
from google.cloud import bigquery
import os
from google.cloud.bigquery.client import Client
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn import preprocessing
from datetime import datetime
import math
from decimal import *


# In[142]:


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'G:/hedden/asu-msba-free-trial-conversion-credentials.json'
bq_client = Client()
client = bigquery.Client()
projectID= 'infusionsoft-looker-poc'
#standardSQLjob_config = bigquery.QueryJobConfig()
##read Usage data from Bigquery, with the trial time all equal to 14 days 
job_config = bigquery.QueryJobConfig()
job_config.use_legacy_sql = False##Standard Sql
query="""SELECT
  U.*,F.trial_date as trial_date
FROM
  `infusionsoft-looker-poc.asu_msba_free_trial_conversion.CONFIDENTIAL_usage_data` AS U JOIN
  (SELECT app_name,trial_date,end_day FROM (SELECT app_name,CAST(trial_date AS date) as trial_date,
  IF(CAST(trial_date AS date)<=CAST(new_customer_date AS date),if(CAST(new_customer_date AS date)>DATE_ADD(CAST(trial_date AS date),INTERVAL 13 DAY),DATE_ADD(CAST(trial_date AS date),INTERVAL 13 DAY),CAST(new_customer_date AS date)),DATE_ADD(CAST(trial_date AS date),INTERVAL 13 DAY)) as end_day
  FROM `infusionsoft-looker-poc.asu_msba_free_trial_conversion.CONFIDENTIAL_free_trail_apps_table`) group by app_name,trial_date,end_day) AS F
  ON U.appname=F.app_name
WHERE
  U.date>=F.trial_date
  AND U.date<=F.end_day
ORDER BY U.appname,U.date"""
read= client.query(query,location='US',job_config=job_config)
UsageData=read.result().to_dataframe()
##make a copy of the USage data
UD=UsageData.copy()



# In[105]:


##get the data of free trail
DFreeTrial= pd.read_gbq('SELECT * FROM asu_msba_free_trial_conversion.CONFIDENTIAL_free_trail_apps_table ',projectID)
## build the target by new_customer_date
DFreeTrial['Target']=DFreeTrial['new_customer_date'].apply(lambda x:0 if pd.isna(x) else 1)
##set the period to 13
timeD=pd.Timedelta(days=13)
##set the largest time of convert to be 30 days
timeIgnore=pd.Timedelta(days=30)
##change the date type to timestamp
DFreeTrial['trial_date']=pd.to_datetime(DFreeTrial['trial_date'])
DFreeTrial['new_customer_date']=pd.to_datetime(DFreeTrial['new_customer_date'])
##build the day of the trail end
DFreeTrial['trail_end_date']=None
##set today as the limit of trail end day
Tnow=datetime.now()
for i in DFreeTrial.index:
    if pd.isna(DFreeTrial.loc[i,'new_customer_date']) or (DFreeTrial.loc[i,'trial_date']+timeD).date()<DFreeTrial.loc[i,'new_customer_date'].date():
        DFreeTrial.loc[i,'trail_end_date']=DFreeTrial.loc[i,'trial_date']+timeD## max use of the traildate
    else:DFreeTrial.loc[i,'trail_end_date']=DFreeTrial.loc[i,'new_customer_date']## set new cus to the trail end
    if DFreeTrial.loc[i,'trail_end_date'].date()+timeIgnore < DFreeTrial.loc[i,'new_customer_date'].date():
        DFreeTrial.loc[i,'Target']=0## change the target value if they expend the period if they convert late
    if DFreeTrial.loc[i,'trial_date'].date()>DFreeTrial.loc[i,'new_customer_date'].date():
        DFreeTrial.loc[i,'trail_end_date']=DFreeTrial.loc[i,'trial_date']+timeD
        DFreeTrial.loc[i,'Target']=0## if the new customer day is early than the trail day, set the trail end to the max,and target to 0
    if DFreeTrial.loc[i]['trail_end_date']>Tnow:DFreeTrial.loc[i]['trail_end_date']=Tnow

##drop the duplicate of freetrail data by the app,trial and trial end date   
DFreeTrial=DFreeTrial.drop_duplicates(subset=['app_name','trial_date','trail_end_date'], keep='first', inplace=False)
DFreeTrial['trail_end_date']=pd.to_datetime(DFreeTrial['trail_end_date'])


# In[300]:


def old(x):##build a columns to show wether it is a old customer or not
    if pd.isnull(x['new_customer_date'])==False:
        TD=x['trial_date'];ND=x['new_customer_date']
        if (TD-ND).days<0:
            return 1
    else:return 0
DFreeTrial['OldCu']=DFreeTrial.apply(old,axis=1)


# In[183]:


##Drop the Outliers in both Freetrial and Usage
droplist=['tl471','wd410','mw416','kw563','ov450','jl500']

UD=UD[~UD['appname'].isin(droplist)]
DFreeTrial=DFreeTrial[~DFreeTrial['app_name'].isin(droplist)]


# In[246]:


UDN=UD.drop_duplicates().copy()


# In[187]:


pandas_profiling.ProfileReport(UD)


# In[247]:


UDN['trial_date']=pd.to_datetime(UDN['trial_date'])
UDN['DatePoint']=UDN['date']-UDN['trial_date']
UDN['DatePoint']=(UDN['DatePoint']/np.timedelta64(1, 'D')).astype('int')


# In[248]:


lens=UDN.shape[0]
print(lens)
Nalist=[]
for col in UDN.columns.tolist():
    if UDN[col].isna().sum()==lens:
        Nalist.append(col)
print(Nalist)
UDN.drop(Nalist,axis=1,inplace=True)
        


# In[249]:


def floatc(x):
    if pd.isna(x):
        return np.nan
    else:
        return float(x)
UDN['invoice_amount']=UDN['invoice_amount'].apply(floatc)


# In[189]:


import pandas_profiling
pandas_profiling.ProfileReport(UDN)


# In[274]:


UDNN[coll].astype(float)


# In[242]:


##See the duplicate row of the same day and same app as different operation from customer, and some them together
Col=UDN.columns.tolist()
SUMD=pd.pivot_table(UDN,index=['appname','trial_date'],values=Col,aggfunc=np.sum)
MAXD=pd.pivot_table(UDN,index=['appname','trial_date'],values=Col,aggfunc='max')
def countna(x):
    per=(x.isna().sum())/(x.shape[0])
    return 1-per

countpercent=UDN.groupby(['appname','trial_date']).apply(countna)


# In[252]:


from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
model=LinearRegression()
UDN.columns.tolist()
coll=[x for x in UDN.columns.tolist() if x not in ['trial_date','date','appname','free_email','date'] ]
anan=SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
UDNN=pd.DataFrame(anan.fit_transform(UDN),index=UDN.index,columns=UDN.columns)
UDNN.head()


# In[298]:


coll=[x for x in UDN.columns.tolist() if x not in ['trial_date','date','appname','free_email','date']]
def trend(df): 
    df=pd.DataFrame(df)
    xtrain=pd.DataFrame()
    xtrain['Num']=list(range(0,df.shape[0]))
    ytrain=df.astype(float)
    model.fit(xtrain,ytrain)
    trend=pd.DataFrame(model.coef_.T,columns=df.columns)
    return trend


# In[299]:


coefd=UDNN.groupby(['appname','trial_date'])[coll].apply(trend).unstack()


# In[306]:


print(coefd)


# In[305]:


Merglist=['app_name','trial_date','new_customer_date','OldCu']
DOC=DFreeTrial[DFreeTrial['OldCu']==1][Merglist]


# In[63]:


FomuDate.sort_values(by=['appname','date'], axis=0)


# In[65]:





# In[66]:




