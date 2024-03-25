# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = None

# This is a placeholder, leave it as None
product = None


# %% tags=["injected-parameters"]
# Parameters
product = {
    "nb": "C:\\Users\\kaitr\\Documents\\GitHub\\spaceship-titanic\\products\\preprocessing.ipynb",
    "encoder": "C:\\Users\\kaitr\\Documents\\GitHub\\spaceship-titanic\\products\\encoder.pkl",
    "train": "C:\\Users\\kaitr\\Documents\\GitHub\\spaceship-titanic\\products\\train.csv",
    "val": "C:\\Users\\kaitr\\Documents\\GitHub\\spaceship-titanic\\products\\val.csv",
    "test": "C:\\Users\\kaitr\\Documents\\GitHub\\spaceship-titanic\\products\\test.csv",
}


# %%
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

encoder = OneHotEncoder(drop = 'first', sparse_output = False)

train.set_index('PassengerId', inplace = True)
test.set_index('PassengerId', inplace = True)

train['Deck'] = [np.nan if pd.isna(cabin) else cabin.split('/')[0] for cabin in train['Cabin']]
train['CabinNumber'] = [np.nan if pd.isna(cabin) else cabin.split('/')[1] for cabin in train['Cabin']]
train['Side'] = [np.nan if pd.isna(cabin) else cabin.split('/')[2] for cabin in train['Cabin']]

test['Deck'] = [np.nan if pd.isna(cabin) else cabin.split('/')[0] for cabin in test['Cabin']]
test['CabinNumber'] = [np.nan if pd.isna(cabin) else cabin.split('/')[1] for cabin in test['Cabin']]
test['Side'] = [np.nan if pd.isna(cabin) else cabin.split('/')[2] for cabin in test['Cabin']]

train['CryoSleep'] = [1 if bool(cryosleep) else 0 for cryosleep in train['CryoSleep']]
train['VIP'] = [1 if bool(vip) else 0 for vip in train['VIP']]

test['CryoSleep'] = [1 if bool(cryosleep) else 0 for cryosleep in test['CryoSleep']]
test['VIP'] = [1 if bool(vip) else 0 for vip in test['VIP']]

train.drop('Cabin', axis = 1, inplace = True)
test.drop('Cabin', axis = 1, inplace = True)

cat_vars = ['HomePlanet', 'Destination', 'Deck', 'Side']

train_cat = train[cat_vars]
test_cat = test[cat_vars]

train_cat = encoder.fit_transform(train_cat)
test_cat = encoder.transform(test_cat)

train_cat = pd.DataFrame(train_cat, columns = encoder.get_feature_names_out())
train_cat.index = train.index

test_cat = pd.DataFrame(test_cat, columns = encoder.get_feature_names_out())
test_cat.index = test.index

train_cat.reset_index(inplace = True)
train.reset_index(inplace = True)

test_cat.reset_index(inplace = True)
test.reset_index(inplace = True)

train = train.merge(train_cat, how = 'left', on = 'PassengerId')
test = test.merge(test_cat, how = 'left', on = 'PassengerId')

train.drop(cat_vars, axis = 1, inplace = True)
test.drop(cat_vars, axis = 1, inplace = True)

train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)

train['Transported'] = [1 if transported else 0 for transported in train['Transported']]

train.dropna(inplace = True)

train = train[sorted(train.columns)]
test = test[sorted(test.columns)]

train, val = train_test_split(train, test_size = 0.2)

with open(product['encoder'], 'wb') as f:

    pkl.dump(encoder, f)

train.to_csv(product['train'], index = False)
val.to_csv(product['val'], index = False)
test.to_csv(product['test'], index = False)
