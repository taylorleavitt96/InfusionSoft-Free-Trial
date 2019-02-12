
# coding: utf-8

# In[3]:


import pandas as pd
from google.cloud import bigquery
import os
from google.cloud.bigquery.client import Client
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn import preprocessing
from datetime import datetime
import math


# In[4]:


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'G:/hedden/asu-msba-free-trial-conversion-credentials.json'
bq_client = Client()
client = bigquery.Client()
projectID= 'infusionsoft-looker-poc'
DFreeTail= pd.read_gbq('SELECT * FROM asu_msba_free_trial_conversion.CONFIDENTIAL_free_trail_apps_table ',projectID)


# In[5]:


def old(x):
    if x.to_pydatetime().year <2017:
        x=1
    else:
        x=0
    return x
DFreeTail['oldcustomer']=None
for i in DFreeTail.index:
    DFreeTail.loc[i,'oldcustomer']=old(DFreeTail.loc[i,'new_customer_date'])
print("The shape of Free trial data is :",DFreeTail.shape)


# In[6]:


print(sum(DFreeTail['oldcustomer']))## get the number of client who are the payment user before 2017


# In[7]:


DFreeTrial=DFreeTail[DFreeTail['oldcustomer']==0]## delete all the user that is old user


# In[8]:


cols=['oldcustomer','in_trial_count_of_launched_campaigns','in_trial_last_campaign_launched_on','in_trial_count_of_published_campaigns','in_trial_last_campaign_published_on','in_trial_count_of_contacts_added','in_trial_count_of_logins','in_trial_total_emails_sent','in_trial_last_email_date','contract_status','cancel_eligible','current_edition_type','initial_edition','current_mrr','initial_mrr','count_of_active_users_in_last_6_months']
DFreeTrial=DFreeTrial.drop(cols,axis=1)


# In[9]:


Head=DFreeTrial.head()
Des=DFreeTrial.describe()
print("The View of the Free trial data is :",Head)


# In[10]:


print("The describe for initial numerical data is:",Des)##see the data contain in numerial feature


# In[11]:


print("The info of the Free trial data is :",DFreeTrial.info())##get the whole information of the dataset


# In[12]:


print("The Unique value in each columns of Free trial data is :",DFreeTail.nunique())
##to se the unique value in the each columns,
##so their must have some duplicate in the appname which should be as big as the dataset,same as trial_date and contact ID


# In[13]:


print("The number of  customer who didn't convert of Free trial data is:",DFreeTail['new_customer_date'].isnull().sum())
## THis is a significant imbalance data


# In[14]:


DFreeTrial['Target']=DFreeTrial['new_customer_date'].apply(lambda x:0 if pd.isna(x) else 1)
timeD=pd.Timedelta(days=13)
timeIgnore=pd.Timedelta(days=30)
DFreeTrial['trial_date']=pd.to_datetime(DFreeTrial['trial_date'])
DFreeTrial['new_customer_date']=pd.to_datetime(DFreeTrial['new_customer_date'])
DFreeTrial['trail_end_date']=None
Tnow=datetime.now()
for i in DFreeTrial.index:
    if pd.isna(DFreeTrial.loc[i,'new_customer_date']) or (DFreeTrial.loc[i,'trial_date']+timeD)<DFreeTrial.loc[i,'new_customer_date']:
        DFreeTrial.loc[i,'trail_end_date']=DFreeTrial.loc[i,'trial_date']+timeD
    else:DFreeTrial.loc[i,'trail_end_date']=DFreeTrial.loc[i,'new_customer_date']
    if DFreeTrial.loc[i,'trail_end_date']+timeIgnore < DFreeTrial.loc[i,'new_customer_date']:
        DFreeTrial.loc[i,'Target']=0
    if DFreeTrial.loc[i,'trial_date']>DFreeTrial.loc[i,'new_customer_date']:
        DFreeTrial.loc[i,'trail_end_date']=DFreeTrial.loc[i,'trial_date']+timeD
        DFreeTrial.loc[i,'Target']=0
    if DFreeTrial.loc[i]['trail_end_date']>Tnow:DFreeTrial.loc[i]['trail_end_date']=Tnow
##get the target value
##(ignore the appname who convert but after more than 30 days and who has been a paying customer befor the trial begin)


# In[15]:


DFreeTrial['trail_end_date']=DFreeTrial['trail_end_date'].astype("datetime64")


# In[16]:


DFreeTrial['Target'].value_counts().plot.bar(title="Target Balance View")
##get the view of the target value


# In[17]:


Categoricallist=[]
for cols in DFreeTrial.columns.tolist():
    if DFreeTrial[cols].dtypes==object:Categoricallist.append(cols)
print(Categoricallist)


# In[18]:


for i,cate in enumerate(Categoricallist):
    DFreeTrial[cate].fillna('MISSING',inplace=True)


# In[19]:


def cleatext(text):
    sw=stopwords.words('english')
    text=text.lower()
    patten1=string.digits
    patten2=string.punctuation
    regex=re.compile(r"[%s%s]"%(patten1,patten2))
    text=regex.sub(" ",text)
    regex=re.compile(r"\s+")
    text=regex.sub(" ",text)
    textlist=text.split()
    textclean=" ".join([w for w in textlist if w.lower() not in sw])
    return textclean.split()
def diff(listA):
    text=['google','infusionsoft','partner','facebook','freetrial','offlinechat','softwareadvice','twitter','onlinechat','email','marketing','forbes','recruiting','demo','MISSING']
    text1=['google','web','facebook','twitter','email','softwareadvice','forbes','doubleyoursales','marketing','www','com','search','emailfooter']
    text2=['infusionsoft','freetrial','demo','salesline','direct','mobile','onlinechat','offlinechat','us','oninechat','content','salesforce','chat','free']
    text3=['partner','referral','recruiting','social','lead']
    text4=['MISSING']
    text5=['marketo', 'campaign']
    output = []
    retA=[]
    for x in listA:
        if x not in output:
            output.append(x)
    for i in output:
        if i in text:
            retA.append(i)
    if retA==[]:
        retA=output
    group='other'
    for i in retA:
        if i in text2: group='InfusionWeb'
        if i in text1: group='Advertise'
        if i in text3: group='Refer'
        if i in text4: group='MISSING'
        if i in text5: group='marketo_campaign'
    return group


# In[20]:


DFreeTrial['lead_lead_source']=DFreeTrial['lead_lead_source'].apply(lambda x:cleatext(x))
DFreeTrial['lead_lead_source']=DFreeTrial['lead_lead_source'].apply(lambda x:diff(x) )
DFreeTrial[DFreeTrial['lead_lead_source']!='MISSING']['lead_lead_source'].value_counts().plot.bar(title='leadsource')


# In[21]:


DFreeTrial['contact_lead_source']=DFreeTrial['contact_lead_source'].apply(lambda x:cleatext(x))
DFreeTrial['contact_lead_source']=DFreeTrial['contact_lead_source'].apply(lambda x:diff(x))
DFreeTrial[DFreeTrial['contact_lead_source']!='MISSING']['contact_lead_source'].value_counts().plot.bar(title='contactsource')


# In[22]:


def datediff(a,b):
    if a.all()!=None and b.all()!=None:
        diff = (a-b).days
    else:
        diff=math.inf
    if a<b:
        diff=-diff
    return diff


# In[23]:


Timelist=[]
for cols in DFreeTrial.columns.tolist():
    if DFreeTrial[cols].dtypes==DFreeTrial['trial_date'].dtypes:Timelist.append(cols)
print(Timelist)


# In[24]:


Timelist2=Timelist[1:]
print(Timelist2)


# In[29]:


i=0;a=len(Timelist)
while i<a-1:
    date1=Timelist[i]
    Timelist2=Timelist[i+1:]
    for date2 in Timelist2:
        name='-'.join([date1,date2])
        DFreeTrial[name]=(DFreeTrial[date1]-DFreeTrial[date2]).dt.days
        
        print(DFreeTrial[name])
    i=i+1


# In[2]:


DFreeTrial.isnull().sum()


# In[32]:


DFreeTrial['opportunityYes']=DFreeTrial['opportunityYes'].apply(lambda x:0 if pd.isna(x) else 1)


# In[41]:


def transfer(x):
    cnew = x.astype('category')
    cnew=cnew.cat.codes
    return cnew
IDlist=['app_name','account_id','opportunity_id','contact_id']
Categoricallist=[]
for cols in DFreeTrial.columns.tolist():
    if DFreeTrial[cols].dtypes==object:Categoricallist.append(cols)
TansferList=[x for x in Categoricallist if x not in IDlist]
print(TansferList)


# In[44]:


for columns in TansferList:
    Nname='_'.join(['Num',columns])
    DFreeTrial[Nname]= transfer(DFreeTrial[columns])
    DFreeTrial[Nname].value_counts().plot.bar(title=Nname)
    plt.show()


# In[48]:


print(DFreeTrial.info())


# In[46]:


f, ax = plt.subplots(figsize=(25, 24))
corr =  DFreeTrial.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)


# In[63]:


DFreeTrial['trial_date']=DFreeTrial['trial_date'].dt.date
DFreeTrial['new_customer_date']=DFreeTrial['new_customer_date'].date
DFreeTrial[DFreeTrial['new_customer_date']<DFreeTrial['trial_date']][['app_name','trial_date','new_customer_date']]

