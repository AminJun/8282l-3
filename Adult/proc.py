#from https://archive.ics.uci.edu/ml/datasets/adult

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

np.random.seed(42)

columns = ["age", "type_employer", "fnlwgt", "education", 
                "education_num","marital", "occupation", "relationship", "race","sex",
                "capital_gain", "capital_loss", "hr_per_week","country", "income"]

data = pd.read_csv('adult.data', names = columns)
test = pd.read_csv('adult.test', names = columns, skiprows=[0]) 

data = data.append(test, ignore_index=True)
del test
import pdb; pdb.set_trace()
data[['age']] = data[['age']].apply(pd.to_numeric)
pdb.set_trace()
data = data[~data.isin(['?'])]
pdb.set_trace()
data = data[~data.isin([' ?'])]
pdb.set_trace()
data = data.dropna()
pdb.set_trace()
data = shuffle(data)
pdb.set_trace()

del data['fnlwgt']
pdb.set_trace()
income_replace_dict = {' >50K.':'>50K', ' <=50K':'<=50K', ' <=50K.':'<=50K', ' >50K':'>50K'}
data = data.replace({'income':income_replace_dict})
pdb.set_trace()
america_etc = [ ' United-States', ' Canada', ' Outlying-US(Guam-USVI-etc)', ' Puerto-Rico']
central_america = [' Guatemala', ' Honduras', ' Nicaragua', ' El-Salvador',' Cuba', ' Mexico', ' Dominican-Republic', ' Jamaica', ' Haiti']
south_america = [' Ecuador', ' Columbia', ' Peru', ' Trinadad&Tobago']
asia = [' Taiwan', ' China', ' Iran', ' Japan', ' Thailand', ' India', ' Philippines', ' Vietnam', ' Laos', ' Hong', ' South', ' Cambodia']

replace_dict = {} 
all_country_values = set(list(data['country'].values))
for c in all_country_values:
    if c in america_etc:
        replace_dict[c] = 'America_Etc'
    elif c in central_america:
        replace_dict[c] = 'Central_America'
    elif c in south_america:
        replace_dict[c] = 'South_America'
    elif c in asia:
        replace_dict[c] = 'Asia'
    else:
        replace_dict[c] = 'Europe'
pdb.set_trace()
data = data.replace({'country':replace_dict})
pdb.set_trace()
data = pd.get_dummies(data)
pdb.set_trace()
del data['income_>50K']
pdb.set_trace()
labels = data['income_<=50K'].copy()
del data['income_<=50K']
pdb.set_trace()
data = data.values.astype('uint8')
pdb.set_trace()
labels = labels.values.astype('uint8')
psb.set_trace()
exit(0)
np.save('data.npy',data)
np.save('labels.npy',labels)
