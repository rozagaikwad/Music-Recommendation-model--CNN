# Music-Recommendation-model--CNN
Designed a model to calculate the probability of the most played song using basic ml principles 

import pandas as pd
df = pd.read_csv('data_recommend.csv')

df.head()
df.info()
# dropping ALL duplicate values
df.drop_duplicates(subset ="song_title", inplace = True)
duplicate = df[df.duplicated(['song_title'])]
 
# Convert the whole dataframe as a string and display
print(duplicate.to_string())
df.shape
# checking for missing value
df.isnull()

df.isna().any().any()
# no missing value found
df.isnull().values.any()

# normalizing data 
from sklearn.preprocessing import MinMaxScaler

df2 = df
df2 = df2.drop(columns=' id')
df2 = df2.drop(columns='song_title')
df2 = df2.drop(columns='artist')


x = df2.values
min_max_scaler =MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

df2 = pd.DataFrame(x_scaled,columns=df2.columns)

# print(df2.to_string())
print(df2)    

# calculating variance of each numeric feature after normalizing song duration and time_signature are found to have very low variance
df2.var()
# degree of spread in your data

# generating lower quartile, upper quartile and inter quartile range for each numeric variable in dataset.
Q1 = df.quantile(0.25)
#print("Q1=",Q1)
Q3 = df.quantile(0.75)
#print("Q3=",Q3)
IQR = Q3 - Q1
print(IQR)

# for numeric outlier removal interquartile range multiplier(k) is set at 1.5
dfx = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
dfx.head()

import seaborn as sns

# correlation between the different parameters(pairwise correlation for all columns)
tc = df2.corr()
 
# plot a heatmap of the correlated data
sns.heatmap(tc)

# dfx is our training data frame-we remove unnecessary features
# loudness are removed because they are highly correlated with energy
# duration is removed because it has very low variance
# artist and song name are unnecessary features
index = dfx[' id']
dfx = dfx.drop(columns=['artist','song_title','loudness','duration_ms',' id','time_signature'],axis=1)
dfx.reset_index(drop=True, inplace=True)
dfx

# normalized data for KNN so that every feature plays an equal role
x = dfx.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfx = pd.DataFrame(x_scaled,columns=dfx.columns)
dfx.head()

# importing KNN 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# custom index
dfx.index = index
dfx.head()
import numpy as np

# weighted KNN
weights=0

while(weights==0):
  weights = np.random.choice(dfx.shape[1])

model_knn = NearestNeighbors(algorithm='brute',metric='wminkowski',metric_params={'w': weights}, p=2, n_jobs=-1)
mat_songs = csr_matrix(dfx.values)

# model training
model_knn.fit(dfx.values)

index=df[' id']
dft=df.drop(columns=['artist','song_title','loudness','duration_ms',' id','time_signature'])
dft.index=index
dft
# recommending songs
def recommend(idx, model, number_of_recommendations):
    query = dft.loc[idx].to_numpy().reshape(1,-1)
    print('Searching for recommendations-')
    distances, indices = model.kneighbors(query,n_neighbors = number_of_recommendations)
    #print(distances+indices)
    
    for i in indices:
        print(df[['song_title','artist']].loc[i].where(df[' id']!=idx))

# Tester
name = input('Enter song title:')
print('\nSearch results:')
print(df[['artist','song_title']].where(df['song_title'] == name).dropna())

ind = int(input('\nEnter the index value of the required song: '))
idx = df[' id'].loc[ind]
song = df['song_title'].loc[ind]
artists = df['artist'].loc[ind]
print('Song selected is ',song, 'by', artists)

nor = int(input('\nEnter number of recommendations: '))
recommend(idx, model_knn, nor)



