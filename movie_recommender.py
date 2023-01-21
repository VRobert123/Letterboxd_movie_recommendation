from joblib import load
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from sklearn.preprocessing import LabelEncoder

#object
class recommender():

    def __init__(self, user):
        self.user = user
        self.link = f'https://letterboxd.com/{self.user}/films/page/'
        #basic dataframe for store the user's data
        self.df = pd.DataFrame()
        self.df['user'] = self.user
        self.df['title'] = 0
        self.df['rating'] = 0
        #load pretrained model
        self.model = load('elements/model.joblib')

        self.i = 1
        while True:
            #download data from letterbox, do while until there are movies on the next page,
            #and put the movie title and ratings to the df dataframe
            self.url = self.link + str(self.i)
            self.response = requests.get(self.url)
            self.soup = BeautifulSoup(self.response.content, "html.parser")
            self.movies = self.soup.find_all("li", {'class': "poster-container"})
            for movie in self.movies:
                rating = movie.text.strip()
                stars = rating.count('★')
                halfs = rating.count('½') * 0.5
                title = movie.find('img')['alt']
                self.df.loc[len(self.df)] = self.user, title, stars + halfs

            #break the loop when there are no more movies left
            if self.movies == []:
                break

            self.i += 1

        #create labelencoder variable and load classes from the elements folder
        self.le = LabelEncoder()
        self.le.classes_ = np.load('elements/classes.npy', allow_pickle=True)
        #encode the titles in df
        self.df['title'] = self.le.transform(self.df['title'])
        #if df contains titles what are not in the pretrained model, delete them
        self.df = self.df[self.df['title'].isin(pd.read_csv('elements/titles.csv', sep=';', index_col=[0])['title'])]
        #create a copy for later tasks
        self.df_copy = self.df.copy()
        #transform df to wide matrix
        self.df = pd.pivot_table(self.df, values='rating', index=['user'], columns='title').reset_index()
        self.df = pd.DataFrame(self.df, columns=pd.read_csv('elements/columns.csv', sep=';')['title']).fillna(0)
        #transform the df
        self.W_new = self.model.transform(self.df)
        #read topic_titles csv (transformed H matrix)
        self.topic_titles = pd.read_csv('elements/topic_titles.csv', sep=';', index_col=[0])
        #create result dataframe, what contains movie titles from the user's category (topic)
        result = self.topic_titles[self.topic_titles['topics'] == list(pd.DataFrame(self.W_new).idxmax(axis=1))[0]]
        #remove movies what the user has been seen
        result = result.merge(self.df_copy, how='outer')
        result = result[result['rating'].isnull()]
        #inverse transform titles from int to string
        result['title'] = self.le.inverse_transform(result['title'])
        #sort values and print titles
        result = result.sort_values('ratio', ascending=False).head(10)
        print(result[['title']])


if __name__ == "__main__":
    #recommender('salty_')
    pass