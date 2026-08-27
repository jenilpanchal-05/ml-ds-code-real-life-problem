import pandas as pd , numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit , cross_val_score 
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

data=pd.read_csv("practicing.csv")
data.head()
data_2=data.copy()

data_copy=data.copy()
data_copy['total_bedrooms']=data_copy['total_bedrooms'].fillna(data['total_bedrooms'].mean())
data_copy.isnull().sum()
data_copy['total_bedrooms'].isnull().sum()

data_2_features=data_2.drop("ocean_proximity",axis=1)
data_2_labels=data_2[['ocean_proximity']]

number_pipeline=Pipeline([
    ['Impute',SimpleImputer(strategy='mean')],
    ['MinMax',MinMaxScaler(feature_range=(-1,1))]
])

features_pipeline=Pipeline([
    ['Encoder',OneHotEncoder(handle_unknown='ignore')]
    
])
full_pipeline=ColumnTransformer([
    ('numbers',number_pipeline,data_2_features.columns),
    ('features',features_pipeline,['ocean_proximity'])
])
converting=full_pipeline.fit_transform(data_2)
dataframing=pd.DataFrame(converting,columns=full_pipeline.get_feature_names_out(),index=data_2.index)
dataframing

# Now the data visualization is present here and we will do 
import matplotlib.pyplot as plt 
dataframing.plot(kind='scatter',x='numbers__latitude',y='numbers__longitude',cmap='inferno',c='numbers__latitude')
plt.grid(True)
plt.title("Housing")
plt.show()