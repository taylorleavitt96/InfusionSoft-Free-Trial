
# coding: utf-8

# In[1]:


import pandas as pd
from google.cloud import bigquery
import os
from google.cloud.bigquery.client import Client
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
from sklearn import preprocessing
from datetime import datetime
import math
from decimal import *


# In[2]:


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'G:/hedden/asu-msba-free-trial-conversion-credentials.json'
bq_client = Client()
client = bigquery.Client()
projectID= 'infusionsoft-looker-poc'


# # The summary of the project

# #The Purpose:The Purpose of this project is to predict the possibility of a freetrial user convert to payment user.
# #The target:Our client and us decide to focus on the customer who covert in a short period after the trial start.
# #The data: The data is extract from BigQuery project. And it includes two tables\n
#           #One is for usage data which contain all the information that how the user manipulate their account(app) 
#           #Another is for trial data which contain the information of the user's inherent attribute to the free trial.

# # First: Start with the Free Trial data

# extract the data

# In[3]:


DFreeTrial=pd.read_gbq('SELECT * FROM asu_msba_free_trial_conversion.CONFIDENTIAL_free_trail_apps_table ',projectID)


# Get the basic imformation of the dataset

# In[4]:


import pandas_profiling
pandas_profiling.ProfileReport(DFreeTrial)


# In[9]:


def isnaS(x):
    if pd.isna(x):
        return 0
    else: return 1
DFK=pd.DataFrame()
for col in DFreeTrial.columns.tolist():
        DFK[col]=DFreeTrial[col].apply(isnaS)


# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[56]:


CORR=pd.DataFrame(DFK.corr()['new_customer_date'])
t=CORR[CORR['new_customer_date']>0.5].index.tolist()


# In[51]:


for col in CORR[CORR['new_customer_date']>0.5].index:
    print(col,':',len(DFK[(DFK[col]==1)&(DFK['new_customer_date']==1)])/len(DFK[DFK[col]==1]))


# From the columns's name we can basic get the representative meaning of it. Because we only want the the information about the behavior of a user during the Free trial, so we can delete all the columns,we are sure about,which contain the promote information for user. So we will only keep ['app_name',
#  'account_id',
#  'opportunity_id',
#  'opportunity_stage_name',
#  'opportunity_owner_name',
#  'opportunity_demo_date',
#  'is_free_trial_initiated',
#  'trial_date',
#  'contact_id',
#  'contact_lead_source',
#  'contact_phase',
#  'lead_lead_source',
#  'lead_converted_date','initial_edition','new_customer_date',
#  'start_date',
#  'kickstart_owner_name',
#  'kickstart_owner_role',
#  'kickstart_type','edition_category','Target',
#  'trail_end_date']
# 

#  We decide to make appname together with trial_date to be the unique id of each record

# In[62]:


DFreeTrial['target']=DFreeTrial['new_customer_date'].apply(isnaS)
x=DFreeTrial.columns.tolist()
colFN=[s for s in x if s not in t]
FreeD=DFreeTrial[colFN]


# In[63]:


print(colFN)


# With the EDA going on, we find that some missing maybe caused by the SQL join work. So We decided to do the opposite. We saperate the dataset into two dataset, drop duplicate, keep the unique value and than merge them together again.

# In[67]:


AP=FreeD[['app_name','is_free_trial_initiated','trial_date','account_id','contact_id','target']]
Data=FreeD.copy()


# In[68]:


def dropDup(df):
    dfN=df.groupby(['app_name','trial_date'])[[cat]].apply(lambda x: x.sort_values([cat],ascending=False).iloc[0,:])
    return dfN
DFL=[]
for cat in ['contact_lead_source', 'contact_phase', 'lead_lead_source', 'lead_converted_date', 'count_of_active_users_in_last_6_months', 'in_trial_first_email_date', 'in_trial_last_email_date', 'in_trial_total_emails_sent', 'in_trial_count_of_logins', 'first_login_on', 'second_login', 'in_trial_count_of_contacts_added', 'first_contact_on', 'last_contact_on', 'in_trial_first_campaign_published_on', 'in_trial_last_campaign_published_on', 'in_trial_count_of_published_campaigns', 'in_trial_first_campaign_launched_on', 'in_trial_last_campaign_launched_on', 'in_trial_count_of_launched_campaigns']:
    dfN=dropDup(Data)
    DFL.append(dfN)
DFree=pd.concat(DFL,axis=1)
DFree['app_name'],DFree['trial_date']=zip(*DFree.index)
AP.drop_duplicates(inplace=True)
FreeDD=AP.merge(DFree,left_on=['app_name','trial_date'],right_on=['app_name','trial_date'])


# In[69]:


pandas_profiling.ProfileReport(FreeDD)


# For the account id , its unique number is smaller than appname and no null, so it means some account may have 2 app name. But this project focus more on every app not account.So we decide to make a columns that can show how many app that an account have

# In[97]:


def Accounttrans(df):
    dfk=df[['app_name','account_id']].drop_duplicates()
    kacc=dfk.pivot_table(index='account_id',values='app_name',aggfunc=len)
    kacc.columns=['Naccount']
    dfn=df.merge(kacc,left_on=['account_id'],right_on=kacc.index)
    return dfn
FreeDN=Accounttrans(FreeDD)


# For the account id, its unique number is bigger than appname and no null, so it means some appname may have more than 1 contactid and oppurtunity id.And nearly all of this two columns are unique. So we decide to make a columns that can show how many account and oppurtunity that an app have

# In[98]:


def Freetrans(df,func,col):
    dfN=df.groupby(['app_name','trial_date'])[col].apply(func)
    for i in range(len(col)): col[i]='N'+col[i]
    dfN.columns=col
    dfN['app_name'],dfN['trial_date']=zip(*dfN.index)
    dfN.reset_index(drop=True, inplace=True)
    df=df.merge(dfN,left_on=['app_name','trial_date'],right_on=['app_name','trial_date'])
    return df
def countnna(x):
    if pd.isna(x.any):
        pre=x.nunique()-1
    else:pre=x.nunique()
    return pre
FreeDN=Freetrans(FreeDN,countnna,['contact_id'])


# In[99]:


FreeDN.drop_duplicates(inplace=True)
FreeDN.head()


# In[100]:


droplist=['tl471','wd410','mw416','kw563','ov450','jl500']
FreeDN=FreeDN[~FreeDN['app_name'].isin(droplist)]


# We decide to extrat information from text data as 'lead_lead_soucre' and 'contact_lead_source', becasue their are so many unique value in it.We make several group to cluster the text basic on their meaning, and we also fill the missing value with unknow

# In[165]:


from sklearn.impute import SimpleImputer
soursenan=SimpleImputer(missing_values=None, strategy='constant', fill_value='unk')
soure=['contact_lead_source','lead_lead_source']
Freesoure=pd.DataFrame(soursenan.fit_transform(FreeDN[soure]),index=FreeDN.index,columns=['N'+x for x in soure])
def soursetrans(text):
    sw=stopwords.words('english')
    text=text.lower()
    patten1=string.digits
    patten2=string.punctuation
    regex=re.compile(r"[%s%s]"%(patten1,patten2))
    text=regex.sub(" ",text)
    regex=re.compile(r"\s+")
    text=regex.sub(" ",text)
    textlist=text.split()
    textclean=[w for w in textlist if w.lower() not in sw]
    text1=['google','web','email','softwareadvice','forbes','doubleyoursales','www','com','emailfooter']
    text2=['infusionsoft','freetrial','demo','direct','content','free','search']
    text3=['partner','referral','recruiting','lead']
    text4=['unk']
    text5=['marketo', 'campaign']
    text6=['salesline','mobile','onlinechat','offlinechat','chat','us','marketing','salesforce']
    text7=['facebook','twitter','social']
    output = []
    retA=[]
    for x in textclean:
        if x not in output:output.append(x)
    for i in output:
        if i in text:retA.append(i)
    if retA==[]:retA=output
    group='other'
    for i in retA:
        if i in text1: group='advertise'
        if i in text2: group='directsearch'
        if i in text3: group='refer'
        if i in text4: group='zunknown'
        if i in text5: group='marketo_campaign'
        if i in text6: group='chat'
        if i in text7: group='socialmedia'
    return group
from nltk.corpus import stopwords
for cat in Freesoure.columns.tolist():
    Freesoure[cat]=Freesoure[cat].apply(lambda x:soursetrans(x))
FreeDN=pd.concat([FreeDN,Freesoure],axis=1)


# In[102]:


##from matplotlib import pyplot as plt
##Freesoure['Nlead_lead_source'].value_counts().plot.bar(title='leadsource')
##Freesoure['Ncontact_lead_source'].value_counts().plot.bar(title='contactsource')


# Timedata is useless in this machinelearning project. We try to built a timedelta to represent the meaning for each timestamp columns.
# First is lead convert time, we use lead convert time to minus trial date, and make the min of it to be -13

# In[140]:


FreeK=FreeDN.copy()
Tnow=datetime.now()
def date_trial(x):
    if x<=-13:return -14
    else:return x
datetype=['in_trial_first_email_date', 'in_trial_last_email_date', 'in_trial_total_emails_sent','lead_converted_date','first_contact_on','in_trial_first_campaign_published_on','in_trial_last_campaign_published_on','in_trial_first_campaign_launched_on','in_trial_last_campaign_launched_on']
for date in datetype:
    FreeK[date]=pd.to_datetime(FreeK[date])
    FreeK[date].fillna(Tnow,inplace=True)
    FreeK[date]=((pd.to_datetime(FreeK['trial_date'].dt.date)-pd.to_datetime(FreeK[date].dt.date))/np.timedelta64(1, 'D')).astype('int')
    FreeK[date]=FreeK[date].apply(date_trial)


# In[141]:


FreeK.head()


# Second we change first cintact on, we choose the most recent first contact to drop the duplicates and use trial_date minus the first contact, fillna with 0 and make the min of it to be -13

# In[132]:


FreeK.drop(['account_id','contact_id','contact_lead_source','lead_lead_source'],inplace=True,axis=1)
FreeK.drop_duplicates(inplace=True)


# In[158]:


FreeK['contact_phase'].unique()


# In[160]:


def phase(x):
    if pd.isna(x) or x=='None': return 0
    else:
        m=x.split(' ')[1]
        return m
        
FreeK['contact_phase']=FreeK['contact_phase'].apply(phase)


# In[162]:


FreeK=FreeK.drop('count_of_active_users_in_last_6_months',axis=1)


# In[164]:


print(FreeK.head())


# In[25]:


pandas_profiling.ProfileReport(FreeDN)


# In[26]:


FreeDN['Target']=FreeDN['new_customer_date'].apply(lambda x:0 if pd.isna(x) else 1)


# In[28]:


DFree
catLE = OrdinalEncoder()
cat=['is_free_trial_initiated','opportunity_stage_name','opportunity_owner_name','contact_phase','Ncontact_lead_source','Nlead_lead_source']
newc=pd.DataFrame(catLE.fit_transform(FreeDN[cat]),index=FreeDN.index,columns=['L'+x for x in cat])
FreeDN=pd.concat([FreeDN,newc],axis=1)
FreeDN.drop(cat,inplace=True,axis=1)


# In[29]:


FreeDN.corr()


# In[30]:


pandas_profiling.ProfileReport(FreeDN)


# # Second Usage Data

# Get the data from Bigquery and limit the periode by trial day

# In[3]:


job_config = bigquery.QueryJobConfig()
job_config.use_legacy_sql = False##Standard Sql
query="""SELECT
  U.*,F.trial_date as trial_date
FROM
  `infusionsoft-looker-poc.asu_msba_free_trial_conversion.CONFIDENTIAL_usage_data` AS U JOIN
  (SELECT app_name,trial_date,end_day FROM (SELECT app_name,CAST(trial_date AS date) as trial_date,
  IF(CAST(trial_date AS date)<CAST(new_customer_date AS date),if(CAST(new_customer_date AS date)>DATE_ADD(CAST(trial_date AS date),INTERVAL 13 DAY),DATE_ADD(CAST(trial_date AS date),INTERVAL 13 DAY),CAST(new_customer_date AS date)),DATE_ADD(CAST(trial_date AS date),INTERVAL 13 DAY)) as end_day
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


# In[4]:


UD[UD['appname']=='jl500']


# In[11]:


appnamebytrialdate = UD.groupby(['appname','trial_date']).size().to_frame('size').reset_index().sort_values(['size'], ascending=[False])


# In[10]:


##Drop the Outliers in both Freetrial and Usage
droplist=['tl471','wd410','mw416','kw563','ov450']

UD=UD[~UD['appname'].isin(droplist)]


# In[14]:


UD.drop_duplicates(inplace=True)


# In[16]:


print(appnamebytrialdate.head())


# In[33]:


pandas_profiling.ProfileReport(UD)


# In[34]:


UD['trial_date']=pd.to_datetime(UD['trial_date'])
UD['date']=pd.to_datetime(UD['date'])
UD['DatePoint']=UD['date']-UD['trial_date']
UD['DatePoint']=(UD['DatePoint']/np.timedelta64(1, 'D')).astype('int')


# In[35]:


UDN=UD.copy()
lens=UDN.shape[0]
Nalist=[]
for col in UDN.columns.tolist():
    if UDN[col].isna().sum()==lens:
        Nalist.append(col)
UDN.drop(Nalist,axis=1,inplace=True)


# In[36]:


def floatc(x):
    if pd.isna(x):
        return np.nan
    else:
        return float(x)
UDN['invoice_amount']=UDN['invoice_amount'].apply(floatc)


# In[37]:


UDN=UDN.fillna(0)


# In[38]:


UDG=UDN[['appname', 'date', 'NUM_CONTACTS', 'NUMEMAILSSENT_AUTO',
       'NUMCONTACTSSENT_AUTO', 'NUMEMAILSSENT_MANUAL',
       'NUMCONTACTSSENT_MANUAL', 'NUMEMAILSSENT_AUTO_SYSTEM',
       'NUMCONTACTSSENT_AUTO_SYSTEM', 'NUMEMAILS_RECEIVED',
       'NUMCONTACTS_RECEIVED', 'NUMEMAILSOPENED_AUTO',
       'NUMCONTACTSOPENED_AUTO', 'NUMEMAILSOPENED_MANUAL',
       'NUMCONTACTSOPENED_MANUAL', 'NUMEMAILSOPENED_AUTO_SYSTEM',
       'NUMCONTACTSOPENED_AUTO_SYSTEM', 'NUMEMAILSOPENED_NULL',
       'NUMCONTACTSOPENED_NULL', 'NUMEMAILSCLICKED_AUTO',
       'NUMCONTACTSCLICKED_AUTO', 'NUMEMAILSCLICKED_MANUAL',
       'NUMCONTACTSCLICKED_MANUAL', 'NUMEMAILSCLICKED_NULL',
       'NUMCONTACTSCLICKED_NULL', 'CONTACTS_UPDATED',
       'PROCESSED_FLOW_ITEM_COUNT', 'LOGIN_COUNT', 'USER_LOGINS',
       'WEBFORM_COUNT', 'WEBFORM_NUM_FORMS', 'WEBFORM_NEW_CONTACTS',
       'WEBFORM_REFERRING_DOMAINS', 'SYSTEM_EMAIL_CLK_COUNT',
       'GOAL_ACHIEVED_COUNT', 'FLOW_RECIPIENT_COUNT', 'CONTACT_GROUP_COUNT',
       'CONTACTS_ADDED_AUTO', 'CONTACTS_ADDED_OTHER',
       'TOTAL_CONTACTS_ADDED_AUTO', 'TOTAL_CONTACTS_ADDED_OTHER',
       'WEB_ANALYTICS_NUM_PAGES', 'WEB_ANALYTICS_NUM_PAGEVIEWS',
       'WEB_ANALYTICS_UNIQUE_VISITORS', 'WEB_ANALYTICS_UNIQUE_CONTACTS',
       'WEB_ANALYTICS_UNIQUE_CUSTOMERS', 'WEB_ANALYTICS_TOTAL_VIEWS',
       'total_processed_usd', 'num_integrations', 'broadcasts_created',
       'campaigns_created', 'funnel_created', 'funnel_published',
       'invoice_amount', 'invoice_created', 'num_invoice_promos',
       'invoices_paid', 'lead_sources', 'actions_created',
       'autotag_config_created', 'merchant_account_created',
       'merchant_infu_created', 'number_of_notes_created',
       'number_of_tasks_created', 'trial_date','DatePoint']]


# In[39]:


UDG.drop_duplicates(inplace=True)


# In[40]:


pt=UDG.groupby(['appname','trial_date']).apply(lambda x: x.sort_values(['DatePoint']))


# In[41]:


import matplotlib.pyplot as plt


# In[42]:


import seaborn as sns


# In[43]:


tar1=FreeDN[FreeDN['Target']==1]['app_name'].tolist()
tar0=FreeDN[FreeDN['Target']==0]['app_name'].tolist()


# In[44]:


import random
t1=random.sample(tar1, 10)
t0=random.sample(tar0, 10)
def exit1(x):
    if x in t1:return True
    else: return False
def exit0(x):
    if x in t0:return True
    else: return False
pt1=pt[pt['appname'].apply(exit1)]
pt0=pt[pt['appname'].apply(exit0)]


# In[48]:



get_ipython().run_line_magic('matplotlib', 'inline')
sns.lineplot(vars=['NUM_CONTACTS', 'NUMEMAILSSENT_AUTO',
       'NUMCONTACTSSENT_AUTO_SYSTEM', 'NUMEMAILS_RECEIVED',
       'NUMCONTACTS_RECEIVED', 'NUMEMAILSOPENED_AUTO', 'NUMEMAILSOPENED_NULL',
       'NUMCONTACTSOPENED_NULL', 'NUMEMAILSCLICKED_MANUAL',
       'CONTACTS_UPDATED',
       'PROCESSED_FLOW_ITEM_COUNT', 'LOGIN_COUNT', 'USER_LOGINS',
       'WEBFORM_COUNT', 'WEBFORM_NEW_CONTACTS',
        'SYSTEM_EMAIL_CLK_COUNT',
       'GOAL_ACHIEVED_COUNT', 
       'TOTAL_CONTACTS_ADDED_AUTO', 'TOTAL_CONTACTS_ADDED_OTHER',
        'WEB_ANALYTICS_TOTAL_VIEWS',
       'total_processed_usd', 'num_integrations', 'broadcasts_created',
       'campaigns_created', 'funnel_created', 'funnel_published',
       'invoice_amount', 'invoice_created', 'lead_sources', 'actions_created',
       'autotag_config_created', 'merchant_account_created',
       'merchant_infu_created', 'number_of_notes_created',
       'number_of_tasks_created'], data=pt1, hue='appname',size=5)
plt.show()


# In[ ]:


sns.lineplot(vars=['NUM_CONTACTS', 'NUMEMAILSSENT_AUTO',
       'NUMCONTACTSSENT_AUTO_SYSTEM', 'NUMEMAILS_RECEIVED',
       'NUMCONTACTS_RECEIVED', 'NUMEMAILSOPENED_AUTO', 'NUMEMAILSOPENED_NULL',
       'NUMCONTACTSOPENED_NULL', 'NUMEMAILSCLICKED_MANUAL',
       'CONTACTS_UPDATED',
       'PROCESSED_FLOW_ITEM_COUNT', 'LOGIN_COUNT', 'USER_LOGINS',
       'WEBFORM_COUNT', 'WEBFORM_NEW_CONTACTS',
        'SYSTEM_EMAIL_CLK_COUNT',
       'GOAL_ACHIEVED_COUNT', 
       'TOTAL_CONTACTS_ADDED_AUTO', 'TOTAL_CONTACTS_ADDED_OTHER',
        'WEB_ANALYTICS_TOTAL_VIEWS',
       'total_processed_usd', 'num_integrations', 'broadcasts_created',
       'campaigns_created', 'funnel_created', 'funnel_published',
       'invoice_amount', 'invoice_created', 'lead_sources', 'actions_created',
       'autotag_config_created', 'merchant_account_created',
       'merchant_infu_created', 'number_of_notes_created',
       'number_of_tasks_created'], data=pt0, hue='appname',size=5)
plt.show()

