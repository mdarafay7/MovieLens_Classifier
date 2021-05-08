#Abdul Rafay Mohammed
#https://dl.acm.org/doi/pdf/10.1145/1454008.1454049
#https://grouplens.org/datasets/movielens/20m/
#https://beckernick.github.io/matrix-factorization-recommender/

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import math
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
links = pd.read_csv("data/links.csv")
tags = pd.read_csv("data/tags.csv")


def R_factorization(R, learning_rate, regulization_factor, K):
    N = len(R)
    M = len(R[0])
    #Initialize P and Q
    P = np.random.rand(N, K)
    Q = np.random.rand(K, M)
    numpy_matrix = R
    pos_id = 0
    for row in R:
        pos_id2 = 0
        #compute eui
        for pos in row:
            R_ui = np.dot(P[pos_id].T ,  Q[:,pos_id2])
            E_ui = R[pos_id][pos_id2] - R_ui
            pos_id2 = pos_id2+1
        pos_id3 = 0
        #update pu, the u-th row of P, and qi, the i-th column of Q, We update the weights in the direction opposite to the gradient
        for pos2 in row:
            P[pos_id] = P[pos_id] + np.dot(learning_rate,(np.dot(E_ui,Q[:,pos_id3]) - np.dot(regulization_factor, P[pos_id])))
            Q[:,pos_id3] = Q[:,pos_id3] + np.dot(learning_rate,(np.dot(E_ui,P[pos_id]) - np.dot(regulization_factor, Q[:,pos_id3])))
            pos_id3 = pos_id3 + 1
        pos_id = pos_id + 1
    print(Q)
    return P,Q

def recommend(R, predictions_df, userID, movies_df, original_ratings_df, num_recommendations):

    preds_df = pd.DataFrame( predictions_df, columns = R.columns)

    # Get and sort the user's predictions
    user_row_number = userID # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending = False)


    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df['userId'] == (userID)]
    user_full = pd.merge(user_data, movies_df, on = "movieId")
    print("Some Movies already rated by user: ",user_full.head(),"\n\n")
    print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print ('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.

    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )
    return user_full, recommendations



main_vector = ratings.pivot(index='userId',columns='movieId',values='rating')
main_vector = main_vector.replace(np.nan, 0)
R = main_vector.values
P, Q = R_factorization(R, 0.0001, 0.02,4)
preds = np.dot(P, Q)

## enter diffrent user ids to recommend movies for that user
already_rated, predictions = recommend(main_vector,preds, 207, movies, ratings, 10)
print(predictions)
