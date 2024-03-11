#!/usr/bin/env python
# coding: utf-8

# In[6]:


#install these packages
get_ipython().system('pip install mlxtend')
get_ipython().system('pip install mlxtend pyfpgrowth')


# In[7]:


#import these packages
import itertools
import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pyfpgrowth


# In[8]:


#To read CSV transaction Files
def load_data(fp):
    txns = []
    with open(fp, 'r') as f:
        reader = pd.read_csv(f)
        for _, r in reader.iterrows():
            txns.append(r['Transaction'].split(', '))
    return txns


# In[9]:


#code implementation for Brute force
def brute_force_search(t_data, s_thresh):
    all_items = set(item for transaction in t_data for item in transaction)
    max_itemset_size = len(all_items)
    freq_sets = {}
    for l in range(1, max_itemset_size + 1):
        current_lvl_itemsets = list(itertools.combinations(all_items, l))
        current_lvl_freq = 0
        for itemset in current_lvl_itemsets:
            frequency = sum(1 for transaction in t_data if set(itemset).issubset(set(transaction)))
            if frequency / len(t_data) >= s_thresh:
                freq_sets[itemset] = frequency
                current_lvl_freq += 1
        if current_lvl_freq == 0:
            break
    return freq_sets


# In[10]:


# Converts a list of transactions into a one-hot encoded DataFrame suitable for itemset mining.
def encode_txns(txns):
    encoder = TransactionEncoder()
    encoded_ary = encoder.fit(txns).transform(txns)
    df_encoded = pd.DataFrame(encoded_ary, columns=encoder.columns_)
    return df_encoded


# In[11]:


#code implementation for apriori Algorithm
def apriori_search(t_data, s_thresh):
    df_encoded = encode_txns(t_data)
    freq_itemsets = apriori(df_encoded, min_support=s_thresh, use_colnames=True)
    return freq_itemsets


# In[12]:


#code implementation for FP-Growth Algorithm
def fpgrowth_search(t_data, s_thresh):
    min_count = int(s_thresh * len(t_data))
    patterns = pyfpgrowth.find_frequent_patterns(t_data, min_count)
    # Constructing the DataFrame with frozenset itemsets and support
    freq_sets = pd.DataFrame([(frozenset(k), v / len(t_data)) for k, v in patterns.items()], columns=['itemsets', 'support'])
    return freq_sets


# In[13]:


def generate_assoc_rules(freq_sets, t_data, c_thresh):
    # No need for `support_only=True` if the DataFrame is structured correctly.
    if not freq_sets.empty:
        rules = association_rules(freq_sets, metric="confidence", min_threshold=c_thresh, support_only = True)
        return rules[['antecedents', 'consequents', 'support', 'confidence']]
    return pd.DataFrame()


# In[14]:


# Compares the performance and output of Brute Force, Apriori, and FP-Growth algorithms on given transaction data.
def algo_comparison(fp, s_thresh, c_thresh):
    t_data = load_data(fp)
    
    # Brute Force
    t_start = time.time()
    bf_sets = brute_force_search(t_data, s_thresh)
    t_bf = time.time() - t_start
    print(bf_sets)
    # Apriori
    t_start = time.time()
    ap_sets = apriori_search(t_data, s_thresh)
    t_ap = time.time() - t_start
    ap_rules = generate_assoc_rules(ap_sets, t_data, c_thresh)
    print(ap_sets)
    # FP-Growth
    t_start = time.time()
    fp_sets = fpgrowth_search(t_data, s_thresh)
    t_fp = time.time() - t_start
    fp_rules = generate_assoc_rules(fp_sets, t_data, c_thresh)
    print(fp_sets)
    # Determine the fastest algorithm
    fastest_time = min(t_bf, t_ap, t_fp)
    fastest_algo = 'Brute Force' if fastest_time == t_bf else ('Apriori' if fastest_time == t_ap else 'FP-Growth')
    
    print(f"Brute Force: {len(bf_sets)} sets in {t_bf:.5f} sec.")
    print(f"Apriori: {len(ap_sets)} sets, {len(ap_rules)} rules in {t_ap:.5f} sec.")
    print(f"FP-Growth: {len(fp_sets)} sets, {len(fp_rules)} rules in {t_fp:.5f} sec.")
    print(f"The fastest algorithm for {fp} is {fastest_algo} with a time of {fastest_time:.5f} sec.\n")
    
    return bf_sets, ap_rules, fp_rules


# In[15]:


def main():
    s_thresh = float(input("Support (fraction): "))
    c_thresh = float(input("Confidence (fraction): "))
    fps = ['Amazon.csv', 'Best_Buy.csv', 'Generic.csv', 'K-Mart.csv', 'Nike.csv']
    for fp in fps:
        print(f"Processing {fp}")
        _, ap_rules, fp_rules = algo_comparison(fp, s_thresh, c_thresh)
        print(f"AP: {len(ap_rules)} rules, FP: {len(fp_rules)} rules\n")

if __name__ == "__main__":
    main()


# In[ ]:




