# -*- coding: utf-8 -*-
"""## Import needed packages

This notebook/script needs pandas and scipy for analysis and boto to access data store on S3.
"""

from datetime import datetime
import pandas as pd
import numpy as np
import xxhash
import re
import os
import gzip
import scipy
import scipy.stats
import s3fs
from google.colab import files
from IPython.display import display  # so we can run this as script as well

"""## Data loading helpers
Define a few helper functions to load and cache data.
"""

# constants for groups
TEST = True
CONTROL = False

# use more memory efficient types for ts,user_id and ab_test_group
def improve_types(df):
    df['ts'] = pd.to_datetime(df['ts'])
    df['ts'] = (df['ts'].astype('int64') / 1e9).astype('int32')
    # preserve the raw user id for export later
    if not export_user_ids:
        df['user_id'] = df['user_id'].apply(xxhash.xxh64_intdigest).astype('int64')
    df['ab_test_group'] = df['ab_test_group'].transform(lambda g: g == 'test')
    return df

def path(audience):
    return "s3://remerge-customers/{0}/uplift_data/{1}".format(customer, audience)

# only keep rows where the event is a revenue event and drop the partner_event
# column afterwards 
def extract_revenue_events(df):
    df = df[df.partner_event == revenue_event]
    return df.drop(columns=['partner_event'])
    
# parquet save and load helper
def to_parquet(df, filename):
    df.to_parquet(filename, engine='pyarrow')

# a little "hack" to convert old file on the fly
def from_parquet_corrected(filename, s3_filename, fs, columns):
    df = from_parquet(filename)
    update_cache = False
    if columns:
      to_drop = list(set(df.columns.values) - set(columns))
      if to_drop:
        df = df.drop(columns=to_drop)
        update_cache = True
        
    # remove events without a user id
    if df['user_id'].dtype == 'object':
      if df[df['user_id'].isnull()].empty == False or df[df['user_id'].str.len() != 36].empty == False:          
        df = df[df['user_id'].str.len() == 36]
        update_cache = True

   
    if df['user_id'].dtype != 'int64':
      df = improve_types(df)
      update_cache = True
    
    if update_cache:
      print(datetime.now(), 'rewritting cached file with correct types (local and S3)', filename, s3_filename)
      to_parquet(df, filename)
      fs.put(filename, s3_filename)
    
    return df

    
def from_parquet(filename):
    return pd.read_parquet(filename, engine='pyarrow')


# helper to download CSV files, convert to DF and print time needed
# caches files locally and on S3 to be reused
def read_csv(audience, source, date, columns=None, chunk_filter_fn=None, chunk_size=10 ** 6):
    now = datetime.now()

    date_str = date.strftime('%Y%m%d')
    
    filename = '{0}/{1}/{2}.csv.gz'.format(path(audience), source, date_str)

    # local cache
    cache_dir = '{0}/{1}/{2}'.format(cache_folder, audience, source)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_filename = '{0}/{1}.parquet'.format(cache_dir, date_str)

    # s3 cache (useful if we don't have enough space on the Colab instance)
    s3_cache_filename = '{0}/{1}/{2}/{3}.parquet'.format(path(audience),
                                                           source, cache_folder, date_str)

    if source == 'attributions':
        cache_filename = '{0}/{1}-{2}.parquet'.format(cache_dir, date_str,
                                                      revenue_event)

        # s3 cache (useful if we don't have enough space on the Colab instance)
        s3_cache_filename = '{0}/{1}/{2}/{3}-{4}.parquet' \
            .format(path(audience), source, cache_folder, date_str, revenue_event)

    fs = s3fs.S3FileSystem(anon=False)
    fs.connect_timeout = 10  # defaults to 5
    fs.read_timeout = 30  # defaults to 15 

    if os.path.exists(cache_filename):
        print(now, 'loading from', cache_filename)
        return from_parquet_corrected(cache_filename, s3_cache_filename, fs, columns)

    
    if fs.exists(path=s3_cache_filename):
        print(now, 'loading from S3 cache', s3_cache_filename)
        
        # Download the file to local cache first to avoid timeouts during the load.
        # This way, if they happen, restart will be using local copies first.
        fs.get(s3_cache_filename, cache_filename)
        
        print(now, 'stored S3 cache file to local drive, loading', cache_filename)
        
        return from_parquet_corrected(cache_filename, s3_cache_filename, fs, columns)

    print(now, 'start loading CSV for', audience, source, date)

    read_csv_kwargs = {'chunksize': chunk_size}
    if columns:
        read_csv_kwargs['usecols'] = columns
        
    df = pd.DataFrame()
    
    if not fs.exists(path=filename):
       print(now, 'WARNING: no CSV file at for: ', audience, source, date, ', skipping the file: ', filename)
       return df
      
    for chunk in pd.read_csv(filename, escapechar='\\', low_memory=False,
                             **read_csv_kwargs):
        if chunk_filter_fn:
            filtered_chunk = chunk_filter_fn(chunk)
        else:
            filtered_chunk = chunk
        
        # we are not interessted in events that do not have a group
        filtered_chunk = filtered_chunk[filtered_chunk['ab_test_group'].isin(['test','control'])]
        # remove events without a user id
        filtered_chunk = filtered_chunk[filtered_chunk['user_id'].str.len() == 36]

        filtered_chunk = improve_types(filtered_chunk)

        df = pd.concat([df, filtered_chunk],
                       ignore_index=True, verify_integrity=True)

    print(datetime.now(), 'finished loading CSV for', date.strftime('%d.%m.%Y'),
          'took', datetime.now() - now)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    print(datetime.now(), 'caching local as parquet', cache_filename)
    to_parquet(df, cache_filename)

    # write it to the S3 cache folder as well
    print(datetime.now(), 'caching on S3 as parquet', s3_cache_filename)
    to_parquet(df, s3_cache_filename)

    return df

"""## DataFrame Helper Functions"""

def calculate_ad_spend(df):
    ad_spend_micros = df[(df.event_type == 'buying_conversion') & (df.ab_test_group == TEST)]['cost_eur'].sum()    
    return ad_spend_micros / 10 ** 6

"""The dataframe created by `marked`  will contain all mark events (without the invalid marks). Remerge marks users per campaign.  If a user was marked once for an audience he will have the same group allocation for consecutive marks (different campaigns) unless manually reset on audience level."""

def marked(df):
    mark_df = df[df.event_type == 'mark']
    
    # we dont need the event_type anymore (to save memory)
    mark_df = mark_df.drop(columns=['event_type'])
    
    mark_df = mark_df[~mark_df['user_id'].isin(double_marked_users.index)]
       
    sorted_mark_df = mark_df.sort_values('ts')
    
    depuplicated_mark_df = sorted_mark_df.drop_duplicates(['user_id'])
    
    return depuplicated_mark_df

"""`merge` joins the marked users with the revenue events and excludes any revenue event that happend before the user was marked."""

def merge(mark_df, attributions_df):
    merged_df = pd.merge(attributions_df, mark_df, on='user_id')
    
    return merged_df[merged_df.ts_x > merged_df.ts_y]

"""## Clean the data
Due to some inconsistencies in the measurement we need to clean the data before analysis.

### Remove duplicated events coming from AppsFlyer
AppsFlyer is sending us two revenue events if they attribute the event to us. One of the events they send us does not contain attribution information and the other one does. Sadly, it is not possible for us to distingish correctly if an event is a duplicate or if the user actually triggered two events with nearly the same information. Therefore we rely on a heuristic. We consider an event a duplicate if the user and revenue are equal and the events are less than a minute apart.
"""

def drop_duplicates_in_attributions(df, max_timedelta):
    sorted = df.sort_values(['user_id', 'revenue'])
    
    # Get values of the previous row
    sorted['last_ts'] = sorted['ts'].shift(1)
    sorted['last_user_id'] = sorted['user_id'].shift(1)
    sorted['last_revenue'] = sorted['revenue'].shift(1)
    
    # Remove rows if the previous row has the same revenue and user id and the ts are less than max_timedelta apart
    filtered = sorted[
        (sorted['user_id'] != sorted['last_user_id']) |
        (sorted['revenue'] != sorted['last_revenue']) |
        ((pd.to_datetime(sorted['ts']) - pd.to_datetime(sorted['last_ts'])) > max_timedelta)]
    
    return filtered[['user_id', 'revenue_eur', 'ts', 'partner_event', 'ab_test_group']]

"""### Remove users with incorrect groups between marks and attributions stream

Some users have conversion/attribution events with the incorrect group (opposing the mark group). Those users need to be removed from the test as well.
"""

def load_users(audience, date, marks, chunk_size=10 ** 6):
    source = 'attributions'
    now = datetime.now()

    date_str = date.strftime('%Y%m%d')
    
    # local cache
    cache_dir = '{0}/{1}/{2}'.format(cache_folder, audience, source)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    filename = '{0}/{1}/{2}.csv.gz'.format(path(audience), source, date_str)


    fs = s3fs.S3FileSystem(anon=False)
    fs.connect_timeout = 10  # defaults to 5
    fs.read_timeout = 30  # defaults to 15 
    
    
    # s3 cache (useful if we don't have enough space on the Colab instance)
    s3_cache_filename = '{0}/{1}/{2}/{3}.users.parquet'.format(path(audience),
                                                           source, cache_folder, date_str)


    cache_filename = '{0}/{1}.users.parquet'.format(cache_dir, date_str)
    
    if os.path.exists(cache_filename):
        print(now, '[users] loading from', cache_filename)
        df = from_parquet(cache_filename)
        df['user_id'] = df['user_id'].astype('int64')
        df['valid'] = df['valid'].astype('bool')
        return df
        

    fs = s3fs.S3FileSystem(anon=False)
    fs.connect_timeout = 10  # defaults to 5
    fs.read_timeout = 30  # defaults to 15 

    if fs.exists(path=s3_cache_filename):
        print(now, '[users] loading from S3 cache', s3_cache_filename)
        
        # Download the file to local cache first to avoid timeouts during the load.
        # This way, if they happen, restart will be using local copies first.
        fs.get(s3_cache_filename, cache_filename)
        
        print(now, '[users] stored S3 cache file to local drive, loading', cache_filename)
        
        return from_parquet(cache_filename)


    print(now, '[users] start loading CSV for', audience, source, date)

    read_csv_kwargs = {'chunksize': chunk_size}
    read_csv_kwargs['usecols'] = ['ts', 'user_id', 'ab_test_group']

    df = pd.DataFrame(columns=['user_id','valid'])
   
    if not fs.exists(path=filename):
       print(now, 'WARNING: no CSV file at for: ', audience, source, date, ', skipping the file: ', filename)
       return df
      
    for chunk in pd.read_csv(filename, escapechar='\\', low_memory=False,
                             **read_csv_kwargs):
        # remove all events that do not have a group
        chunk = chunk[chunk['ab_test_group'].isin(['test','control'])]
        chunk = improve_types(chunk)
        merged_df = merge(marks, chunk)
        invalid = merged_df[merged_df.ab_test_group_x != merged_df.ab_test_group_y]

        invalid = invalid[['user_id']].copy()
        invalid['valid'] = False
        
        
        df = pd.concat([invalid,df],
                       ignore_index=True,
                       verify_integrity=True)
        
        valid = chunk[['user_id']].copy()
        valid['valid'] = True

        df = pd.concat([df,valid], ignore_index=True, verify_integrity=True)
        
        df = df.drop_duplicates(['user_id'])
       
        
    print(datetime.now(), '[users] finished loading CSV for', date.strftime('%d.%m.%Y'),
          'took', datetime.now() - now)
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    to_parquet(df, cache_filename)

    # write it to the S3 cache folder as well
    print(datetime.now(), '[users] caching as parquet', s3_cache_filename)

    to_parquet(df, s3_cache_filename)
    
    return df

"""## Uplift Calculation
We calculate the incremental revenue and the iROAS in line with the [remerge whitepaper](https://drive.google.com/file/d/1PTJ93Cpjw1BeiVns8dTcs2zDDWmmjpdc/view). Afterwards we run a [chi squared test](https://en.wikipedia.org/wiki/Chi-squared_test) on the results to test for significance of the results, comparing conversion to per group uniques.
"""
def uplift(bids_df, attributions_df, index_name, m_hypothesis=1, invalid_users=None, test_inactivity_bias=None, control_inactivity_bias=None):
    # CORRECTION: remove invalid users if any
    test_invalid_users = 0
    control_invalid_users = 0
    if invalid_users is not None:
        #memorize how many invalid users we are removing per group
        invalid_bids_index = bids_df['user_id'].isin(invalid_users)
        invalid_bids = bids_df[invalid_bids_index]      
        test_invalid_users = invalid_bids[invalid_bids['ab_test_group'] == TEST]['user_id'].nunique()
        control_invalid_users = invalid_bids[invalid_bids['ab_test_group'] == CONTROL]['user_id'].nunique()

        # remove invalid users from the main dfs
        bids_df = bids_df[~invalid_bids_index]      
        attributions_df = attributions_df[~attributions_df['user_id'].isin(invalid_users)]

    
    # filter for mark events
    marks_df = marked(bids_df)
        
    # calculate group sizes
    test_group_size = marks_df[marks_df['ab_test_group'] == TEST]['user_id'].nunique()
    if test_group_size == 0:
        print("WARNING: No users marked as test for ", index_name, 'skipping.. ')
        return None
    
    control_group_size = marks_df[marks_df['ab_test_group'] == CONTROL]['user_id'].nunique()
    if control_group_size == 0:
        print("WARNING: No users marked as control for ", index_name, 'skipping.. ')
        return None

    # CORRECTION: compensate for inactivity bias since there are some unobserved invalid users
    # reasoning: we can observed the invalid users only when they have any attribution activity
    # hence we must compensate for invalid users which were inactive (=not seen in attribution stream)
    if invalid_users is not None:
        if test_inactivity_bias is not None:
            test_missing_invalid_correction = 1 / (1 - test_inactivity_bias) - 1
            test_group_size -= test_invalid_users * test_missing_invalid_correction
        if control_inactivity_bias is not None:
            control_missing_invalid_correction = 1 / (1 - control_inactivity_bias) - 1
            control_group_size -= control_invalid_users * control_missing_invalid_correction
    

    # Dask based join, for later and bigger datasets
    # marks_df = dd.from_pandas(marks_df, npartitions=10, sort=True)    
    # attributions_df = dd.from_pandas(attributions_df, npartitions=20, sort=True)
    # merged_df = dd.merge(attributions_df, marks_df, on='user_id')
    # merged_df = merged_df[merged_df.ts_x > merged_df.ts_y]
    # merged_df = merged_df.compute()
        
    # join marks and revenue events    
    merged_df = merge(marks_df, attributions_df)
    grouped_revenue = merged_df.groupby(by='ab_test_group_y')

    # init all KPIs with 0s first:
    test_revenue_micros = 0
    test_conversions = 0
    test_converters = 0

    control_revenue_micros = 0
    control_conversions = 0
    control_converters = 0

    # we might not have any events for a certain group in the time-period,
    if TEST in grouped_revenue.groups:
        test_revenue_df = grouped_revenue.get_group(TEST)
        test_revenue_micros = test_revenue_df['revenue_eur'].sum()
        # test_conversions = test_revenue_df['partner_event'].count()
        # as we filtered by revenue event and dropped the column we can just use
        test_conversions = test_revenue_df['user_id'].count()
        test_converters = test_revenue_df['user_id'].nunique()

    if CONTROL in grouped_revenue.groups:
        control_revenue_df = grouped_revenue.get_group(CONTROL)
        control_revenue_micros = control_revenue_df['revenue_eur'].sum()
        # control_conversions = control_revenue_df['partner_event'].count()
        # as we filtered by revenue event and dropped the column we can just use
        control_conversions = control_revenue_df['user_id'].count()
        control_converters = control_revenue_df['user_id'].nunique()


    # calculate KPIs
    test_revenue = test_revenue_micros / 10 ** 6
    control_revenue = control_revenue_micros / 10 ** 6

    ratio = float(test_group_size) / float(control_group_size)
    scaled_control_conversions = float(control_conversions) * ratio
    scaled_control_revenue_micros = float(control_revenue_micros) * ratio
    incremental_conversions = test_conversions - scaled_control_conversions
    incremental_revenue_micros = test_revenue_micros - scaled_control_revenue_micros
    incremental_revenue = incremental_revenue_micros / 10 ** 6
    incremental_converters = test_converters - control_converters * ratio

    # calculate the ad spend        
    ad_spend = calculate_ad_spend(bids_df)  

    iroas = incremental_revenue / ad_spend
    icpa = ad_spend / incremental_conversions
    cost_per_incremental_converter = ad_spend / incremental_converters
    
    rev_per_conversion_test = 0
    rev_per_conversion_control = 0
    if test_conversions > 0:
        rev_per_conversion_test = test_revenue / test_conversions
    if control_conversions > 0:
        rev_per_conversion_control = control_revenue / control_conversions

    test_cvr = test_conversions / test_group_size
    control_cvr = control_conversions / control_group_size

    uplift = 0
    if control_cvr > 0:
        uplift = test_cvr / control_cvr - 1

    # calculate statistical significance
    control_successes, test_successes = control_conversions, test_conversions
    if use_converters_for_significance or max(test_cvr, control_cvr) > 1.0:
        control_successes, test_successes = control_converters, test_converters
    chi_df = pd.DataFrame({
        "conversions": [control_successes, test_successes],
        "total": [control_group_size, test_group_size]
    }, index=['control', 'test'])
    # CHI square calculation will fail with insufficient data
    # Fallback to no significance
    try:
        chi, p, *_ = scipy.stats.chi2_contingency(
            pd.concat([chi_df.total - chi_df.conversions, chi_df.conversions], axis=1), correction=False)
    except:
        chi, p = 0, 1.0
        
    # bonferroni correction with equal weights - if we have multiple hypothesis:
    # https://en.wikipedia.org/wiki/Bonferroni_correction
    significant = p < 0.05/m_hypothesis

    dataframe_dict = {
        "ad spend": ad_spend,
        "total revenue": test_revenue + control_revenue,
        "test group size": test_group_size,
        'test invalid users': test_invalid_users,
        "test conversions": test_conversions,
        "test converters": test_converters,
        "test revenue": test_revenue,
        "control group size": control_group_size,
        'control invalid users': control_invalid_users,
        "control conversions": control_conversions,
        "control_converters": control_converters,
        "control revenue": control_revenue,
        "ratio test/control": ratio,
        "control conversions (scaled)": scaled_control_conversions,
        "control revenue (scaled)": scaled_control_revenue_micros / 10 ** 6,
        "incremental conversions": incremental_conversions,
        "incremental converters": incremental_converters,
        "incremental revenue": incremental_revenue,
        "rev/conversions test": rev_per_conversion_test,
        "rev/conversions control": rev_per_conversion_control,
        "test CVR": test_cvr,
        "control CVR": control_cvr,
        "CVR Uplift": uplift,
        "iROAS": iroas,
        "cost per incr. converter": cost_per_incremental_converter,
        "iCPA": icpa,
        "chi^2": chi,
        "p-value": p,
        "significant": significant
    }

    # show results as a dataframe
    return pd.DataFrame(
        dataframe_dict,
        index=[index_name],
    ).transpose()

"""### Calculate and display uplift report for the data set as a whole
This takes the whole data set and calculates uplift KPIs.
"""

def uplift_report(bids_df, attributions_df, groups, per_campaign_results, invalid_users=None, test_inactivity_bias=None, control_inactivity_bias=None):
    # calculate the total result:
    report_df = uplift(bids_df, attributions_df, "total",1, invalid_users, test_inactivity_bias, control_inactivity_bias)

    # if there are groups filter the events against the per campaign groups and generate report
    if len(groups) > 0:
        for name, campaigns in groups.items():
            group_bids_df = bids_df[bids_df.campaign_id.isin(campaigns)]
            report_df[name] = uplift(group_bids_df, attributions_df, name, len(groups), invalid_users, test_inactivity_bias, control_inactivity_bias)
            
    if per_campaign_results:
        campaigns = bids_df['campaign_name'].unique()
        for campaign in campaigns:
            name = "c_{0}".format(campaign)
            campaign_bids_df = bids_df[bids_df.campaign_name == campaign]
            report_df[name] = uplift(campaign_bids_df, attributions_df, name, len(campaigns), invalid_users, test_inactivity_bias, control_inactivity_bias)
    return report_df