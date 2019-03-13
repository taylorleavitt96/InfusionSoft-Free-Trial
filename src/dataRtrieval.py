#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:30:10 2019

@author: shravanamee
"""

import datetime
import gc
import os
import numpy as np
from google.cloud import bigquery
import google.auth
import pandas as pd


client = bigquery.Client()



sqlbigquery_data = """SELECT u.* FROM `infusionsoft-looker-poc.asu_msba_free_trial_conversion.CONFIDENTIAL_usage_data` u
join `infusionsoft-looker-poc.asu_msba_free_trial_conversion.CONFIDENTIAL_free_trail_apps_table`  a
on u.appname = a.app_name 
where u.appname not in ('tl471','wd410','mw416','kw563','ov450') and u.date BETWEEN DATE (a.trial_date, "America/Los_Angeles") and 
IF(date_diff(Date(new_customer_date, "America/Los_Angeles"), Date(trial_date, "America/Los_Angeles"),day)<=13,date(a.new_customer_date,"America/Los_Angeles") ,DATE_ADD( date(a.trial_date,"America/Los_Angeles"), interval 13 DAY))"""


queryparam_1 = "select * from asu_msba_free_trial_conversion.CONFIDENTIAL_usage_data u where u.appname in unnest(@appnames) and u.date between (@startdates) and (@enddates)"


# sqlbigquery2= "select * from asu_msba_free_trial_conversion.CONFIDENTIAL_free_trail_apps_table"
#
#
usagedatasetfree= pd.read_gbq(sqlbigquery_data,
                project_id='infusionsoft-looker-poc',
                dialect='standard'
                )


print(len(usagedatasetfree))


# reading free trial app data

free_Trial_app_data = pd.read_csv(r'/Users/shravanamee/Desktop/Free-Trial-Conversion-New/infusionsoft-free-trial/src/data/CONFIDENTIAL_free_trial_apps_000000000000.csv')


head=free_Trial_app_data.head()

print(head)




#converting to date types and adding target


free_Trial_app_data['Target'] = list(map((lambda x: 0 if pd.isna(x) else 1),free_Trial_app_data['new_customer_date']))

free_Trial_app_data['trial_date'] = pd.to_datetime(free_Trial_app_data['trial_date'])
free_Trial_app_data['new_customer_date'] = pd.to_datetime(free_Trial_app_data['new_customer_date'])

print((free_Trial_app_data['trial_end_date'] - free_Trial_app_data['trial_date']))

# calculating trial end date

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


free_Trial_app_data['trial_end_date'] = free_Trial_app_data.apply(lambda row: datediff(row['trial_date'], row['new_customer_date']), axis=1)

# for index,row in free_Trial_app_data.iterrows():
#     row['trial_end_date'] = datediff(row['trial_date'], row['new_customer_date'])

#print(free_Trial_app_data['trial_date']-free_Trial_app_data['trial_end_date'])

    #
    # if row['app_name'] not in appanddatesdict.keys():
    #     appanddatesdict[row['app_name']] = {}
    #     appanddatesdict[row['app_name']]['trialstartdate'] = row['trial_date']
    #     appanddatesdict[row['app_name']]['trialenddate'] = row['trial_end_date']
    #     appanddatesdict[row['app_name']]['index'] = 0
    # else:
    #
    #     appanddatesdict[row['app_name']]['index'] = appanddatesdict[row['app_name']]['index']+1
    #     appname = row['app_name']+'_'+str(appanddatesdict[row['app_name']]['index'])
    #     appanddatesdict[appname]['trialstartdate'] = row['trial_date']
    #     appanddatesdict[appname]['trialenddate'] = row['trial_end_date']



    #usagedatasetfinal=pd.DataFrame()
free_Trial_app_data['trial_date']=free_Trial_app_data['trial_date'].dt.date
free_Trial_app_data['trial_end_date']=free_Trial_app_data['trial_end_date'].dt.date



print(free_Trial_app_data['trial_end_date'].dtype,free_Trial_app_data['trial_date'].dtype)

free_Trial_app_data.to_csv(r'/Users/shravanamee/Desktop/Free-Trial-Conversion-New/infusionsoft-free-trial/src/data/free_Trial_app_data.csv')

query_params = [
    bigquery.ArrayQueryParameter(
        'appnames', 'STRING', ['ac505','ac506']),
    bigquery.ArrayQueryParameter(
        'startdates', 'DATE', free_Trial_app_data['trial_date']),
    bigquery.ArrayQueryParameter(
        'enddates', 'DATE', free_Trial_app_data['trial_end_date'])
]
job_config = bigquery.QueryJobConfig()
job_config.query_parameters = query_params
query_job = client.query(
    queryparam_1,
    location='US',
    job_config=job_config)

usagedata = query_job.to_dataframe()


usagedata['date'] = pd.to_datetime(usagedata['date'])
# usagedata1 = usagedata[(usagedata['appname'] == free_Trial_app_data['app_name']) and (free_Trial_app_data['trial_date'] <= usagedata['date'] <= free_Trial_app_data['trial_end_date'])]



#usagedataset['date'] = pd.to_datetime(usagedataset['date'])
#trialdate = free_Trial_app_data_1app['trial_date']
#enddate = free_Trial_app_data_1app['trial_end_date']


#usagedataset1 = usagedataset.loc[usagedataset.date >= free_Trial_app_data_1app.trial_date, :]





print(free_Trial_app_data['trial_end_date'])










