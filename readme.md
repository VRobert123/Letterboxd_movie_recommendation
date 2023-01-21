# Letterboxd movie recommendation with NMF
### This is a source for a basic movie recommendation system, using Nonnegative Matrix Factorialization and a 1.4M length dataframe, scraped from letterboxd.com

## Usage:
Copy the username from the website:

From:
https://letterboxd.com/USERNAME/ 

Then insert the username for the right place into the movie_recommender.py:

> recommender('USERNAME')

The console will print ten recommended movies for the chosen user.

Please keep in mind, you have to download and keep the 'elements' directory, because there are important files in it for the program.

If you want to regenerate the model, you can use the model.py file with the .csv file in the data directory or you can try another datasets.

## How does it work?

We have a basic dataset with three columns: user, movie title, rating. For a user there are more titles and all titles have a rating. We can transform this dataset to a wide matrix, where every row means an user, and every column means a movie title. The values are the rating scores or NAs (replaced with 0). This is the matrix V, and let the matrix V equals to W x H. We determine a k number, what will mean categories, clusters, topics, depends on the aim. In the W matrix, every row is a user and every column is k (cluster). In the H matrix every row is k and every column is a title. For the better results, we want to minimze the value of V - WH. We can use the NMF model from sklearn for this reason. 

![NMF visual](https://upload.wikimedia.org/wikipedia/commons/f/f9/NMF.png)

Source: Wikipedia

After we got the final W and H matrix, we can analyze them, or there is a possibility, to calculate the k for a new user. This is how the recommendation part is working: when the user types the username to the exact part and runs the script, the user's letterboxd profile became downloaded and transformed to the appropriate form. With the pretrained model, we can calculate the most representative k for the user, and show the certain k's most characteristic movie titles, excluding those already watched by the user. 


## Dependencies

numpy, pandas, bs4, requests, scikit-learn, joblib

## Upgrades

You can expect upgrades after a while.

## License

MIT license.


