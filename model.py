import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import numpy as np

#read data
data = pd.read_csv('data/letterbox_anonym.csv', sep=';')

#Encoding titles from string to int
le = LabelEncoder()
data['title'] = le.fit_transform(data['title'])
data['title'].to_csv('elements/titles.csv', sep=';')
print('Titles saved')

#create V matrix (rows : user, columns: titles)
data_pivot = pd.pivot_table(data, values = 'rating', index=['user'], columns = 'title').reset_index()
data_pivot = data_pivot.fillna(0)
data_pivot = data_pivot.set_index('user')
print(pd.DataFrame(data_pivot.columns))
pd.DataFrame(data_pivot.columns).to_csv('elements/columns.csv', sep=';')
print('Columns saved')

#Use sklearn NMF model, W (rows: user, cols : categories or topics), H (rows: categories or topics, columns: titles)
model = NMF(n_components=10, init='random', random_state=0)
W = model.fit_transform(data_pivot)
H = model.components_
print(f'Model created')

#from W and H create pandas dataframe
user_topics = pd.DataFrame(data= W, index= data_pivot.index)
topics_titles = pd.DataFrame(H)


#create tidy dataframe from H (cols: topics, titles, ratio)
titles=list(topics_titles.columns)
topic_titles = pd.melt(topics_titles, value_vars=titles,value_name='titles', ignore_index=False).reset_index()
topic_titles = topic_titles.rename(columns={"index": "topics", "variable": "title", "titles":"ratio"})
topic_titles.to_csv('elements/topic_titles.csv', sep=';')

#create tidy dataframe from W (cols: userId, topic, ratio)
topics = list(user_topics.columns)
user_topics2 = pd.melt(user_topics, value_vars=topics,value_name='titles', ignore_index=False).reset_index()
user_topics2 = user_topics2.rename(columns={"index": "userId", "variable": "topic", "titles":"ratio"})

#create grouped data to count users in topics
grouped = user_topics2.groupby('user')['ratio'].max().reset_index()
new_df = pd.merge(grouped, user_topics2,  how='left', left_on=['user','ratio'], right_on = ['user','ratio'])

#print
print(new_df.topic.value_counts())
dump(model, 'elements/model.joblib')
np.save('elements/classes.npy', le.classes_)



