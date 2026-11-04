import pandas as pd


# =================================load training data and generate user-item matrix ===================================

# Load training data
df_movies = pd.read_csv("trainingSet.csv")
print("df_movies =")
print(df_movies)

# Use pivot_table to generate a user-item matrix:
# rows = users, columns = movies, values = ratings (0 means not rated)
df_user_item = pd.pivot_table(
    df_movies,
    columns=["movieId"],
    index=["userId"],
    values="rating",
    fill_value=0
)
print("df_user_item =")
print(df_user_item)


# ====================================generate cosine similarity matrix for Users ============================================

# Obtain the userId index of the user-item matrix
indexs = df_user_item.index.tolist()
df_Similarity = pd.DataFrame(index=indexs, columns=indexs, dtype=float)

# Calculate cosine similarity between every pair of users
for i in range(df_Similarity.shape[0]):
    for j in range(df_Similarity.shape[1]):
        sim_numerator = sum(df_user_item.iloc[i, :] * df_user_item.iloc[j, :])
        sim_denominator = (
            sum(df_user_item.iloc[i, :] ** 2) ** 0.5
        ) * (
            sum(df_user_item.iloc[j, :] ** 2) ** 0.5
        )

        if sim_denominator == 0:
            df_Similarity.iloc[i, j] = 0
        else:
            df_Similarity.iloc[i, j] = sim_numerator / sim_denominator

# Set the similarity of each user with himself/herself to 0
for i in range(df_Similarity.shape[0]):
    df_Similarity.iloc[i, i] = 0

print("df_Similarity =")
print(df_Similarity)


# ======================load testing data and conduct user-to-user collaborative filtering=====================================

# Load testing data
df_test = pd.read_csv("testingSet.csv")

# Predict the ratings of user 55
userX = 55
df_userX = df_test[df_test.userId == userX].copy()

# Insert a numeric column to store the estimated ratings
df_userX.insert(loc=3, column="rating_estimate", value=0.0)
print("df_userX =")
print(df_userX)

# Use all 100 users as neighbors (k = 100)
k = 100

for j in range(df_userX.shape[0]):
    movieID_toEst = df_userX.iloc[j, 1]
    print("movieID_to_estimate = %s" % movieID_toEst)

    # 1. Sum the similarities of users who rated this movie.
    # Users who did not rate this movie (rating = 0) must be excluded.
    sum_similarity = 0.0
    for neighborV in range(1, k + 1):
        if df_user_item.loc[neighborV, movieID_toEst] > 0:
            sum_similarity += df_Similarity.loc[userX, neighborV]

    print("sum_similarity = %s" % sum_similarity)

    # 2. Calculate user 55's average rating, excluding unrated movies (zeros).
    userX_ratings = df_user_item.loc[userX]
    userX_score_avg = userX_ratings[userX_ratings > 0].mean()

    print("userX_score_avg = %s" % userX_score_avg)

    # 3. Calculate the weighted deviations from each neighbor's average rating.
    # Only include neighbors who rated the movie being predicted.
    sum_numerator = 0.0
    for neighborV in range(1, k + 1):
        if df_user_item.loc[neighborV, movieID_toEst] > 0:
            neighborV_ratings = df_user_item.loc[neighborV]
            neighborV_score_avg = neighborV_ratings[neighborV_ratings > 0].mean()

            sum_numerator += (
                df_Similarity.loc[userX, neighborV]
                * (df_user_item.loc[neighborV, movieID_toEst] - neighborV_score_avg)
            )

    print("sum_numerator = %s" % sum_numerator)

    # 4. Calculate the predicted rating.
    # If no neighbor has positive similarity, use user 55's average rating.
    if sum_similarity > 0:
        rating_estimate = userX_score_avg + sum_numerator / sum_similarity
    else:
        rating_estimate = userX_score_avg

    df_userX.loc[df_userX.index[j], "rating_estimate"] = rating_estimate

print("Predictions for user 55 =")
print(df_userX[["userId", "movieId", "rating", "rating_estimate"]])


# ===============================Evaluate the Estimation Using RMSE==========================================

RMSE = (
    ((df_userX["rating_estimate"] - df_userX["rating"]) ** 2).mean()
) ** 0.5

print("RMSE = %s" % RMSE)
