import sys
import os
import pandas as pd
import scipy.stats
import math

################################################################################
# average
################################################################################
def avgStat(TrainingFile, movie_id):
    movieDB = pd.read_table(filepath_or_buffer = TrainingFile, delimiter = '\t', names = ['user_ID', 'movies', 'ratings', 'times'])
    ratingTotal = 0.0
    ratingCount = 0.0
    if movie_id >= 0 and movie_id <= 1682:
        for row in movieDB.iloc[0][1:-1]:
            arr = movieDB[movieDB['movies'] == movie_id]
        for row in arr.iterrows():
            ratingTotal = ratingTotal + float(row[1].loc['ratings'])
            ratingCount = ratingCount + 1.0
        if ratingCount == 0:
            return None
        else:
            prediction = ratingTotal/ratingCount
            return prediction

################################################################################
# euclid
################################################################################
def euclidDistance(mainUsrRating, otherUsrRating):
    distance = None
    for rating in otherUsrRating.iterrows():
        if rating[1].loc['movies'] in mainUsrRating:
            if distance is None:
                distance = (rating[1].loc['ratings'] - mainUsrRating[rating[1].loc['movies']])**2
            else:
                distance += (rating[1].loc['ratings'] - mainUsrRating[rating[1].loc['movies']])**2
    return distance

def euclidRatingStat(movieDB, movie_id, similarUsrs, k, weightedTotal, weights):
    counter = 0

    for key, value in sorted(similarUsrs.items(), key = lambda pair: pair[1], reverse=True):
        listOfUsrRatings = movieDB.loc[(movieDB['user_ID'] == key) & (movieDB['movies'] == int(movie_id))]
        for row in listOfUsrRatings.iterrows():
            weightedTotal = weightedTotal + (1.0/(1.0 + value)) * row[1].loc['ratings']
            weights = weights + (1.0/(1.0 + value))

        counter = counter + 1
        if counter >= k and k != 0:
            break

    if weights == 0.0:
        return None
    else:
        prediction = weightedTotal / weights
        return prediction

################################################################################
# pearson
################################################################################
def pearsonCorrelation(mainUsrRating, otherUsrRating):
    mainUser = []
    nextUser = []
    for rating in otherUsrRating.iterrows():
        if rating[1].loc['movies'] in mainUsrRating:
            mainUser.append(mainUsrRating[rating[1].loc['movies']])
            nextUser.append(rating[1].loc['ratings'])
    if math.isnan(scipy.stats.pearsonr(mainUser, nextUser)[0]):
        return None
    else:
        correlation = scipy.stats.pearsonr(mainUser, nextUser)[0]
        return correlation

def pearsonRatingStat(movieDB, movie_id, similarUsrs, k, weightedTotal, weights):
    counter= 0

    for key, value in sorted(similarUsrs.items(), key = lambda pair: pair[1], reverse=True):
        listOfUsrRatings = movieDB.loc[(movieDB['user_ID'] == key) & (movieDB['movies'] == int(movie_id))]
        for row in listOfUsrRatings.iterrows():
            weightedTotal = weightedTotal + (value * ((2.0*((row[1].loc['ratings']) - 1.0) - 4.0) / 4.0))
            weights = weights + abs(value)

        counter = counter + 1
        if counter >= k and k != 0:
            break

    if weights == 0.0:
        return None
    else:
        prediction = weightedTotal / weights
        prediction = 0.5 * ((prediction + 1.0) * 4) + 1
        return prediction

################################################################################
# cosine
################################################################################
def cosineSimilarity(mainUsrRating, otherUsrRating):
    mainUserVector = 0.0
    nextUserVector = 0.0
    cosSumTop = 0.0
    for rating in otherUsrRating.iterrows():
        if rating[1].loc['movies'] in mainUsrRating:
            cosSumTop = cosSumTop + (rating[1].loc['ratings'] * mainUsrRating[rating[1].loc['movies']])
            mainUserVector = mainUserVector + mainUsrRating[rating[1].loc['movies']]**2
            nextUserVector = nextUserVector + rating[1].loc['ratings']**2
    cosDivisor = math.sqrt(mainUserVector) + math.sqrt(nextUserVector)
    if cosDivisor <= 0.0:
        return None
    else:
        cosSim = cosSumTop / cosDivisor
        return cosSim

def cosineRatingStat (movieDB, movie_id, similarUsrs, k, weightedTotal, weights):
    counter = 0

    for key, value in sorted(similarUsrs.items(), key = lambda pair: pair[1], reverse=True):
        listOfUsrRatings = movieDB.loc[(movieDB['user_ID'] == key) & (movieDB['movies'] == int(movie_id))]
        for row in listOfUsrRatings.iterrows():
            weightedTotal = weightedTotal + (value * row[1].loc['ratings'])
            weights = weights + value

        counter = counter + 1
        if counter >= k and k != 0:
            break

    if weights == 0.0:
        return None
    else:
        prediction = weightedTotal / weights
        return prediction

################################################################################
# main
################################################################################

if len(sys.argv) == 6 or len(sys.argv) == 7:
    command = sys.argv[1]

    ############################################################################
    # predict
    ############################################################################
    if command == "predict" and len(sys.argv) == 7:
        command = sys.argv[1]
        TrainingFile = sys.argv[2]
        k = int(sys.argv[3])
        if (k < 0):
            print("Invalid 'k' value!")
            quit()
        algorithm = sys.argv[4]
        user_id = int(sys.argv[5])
        movie_id = int(sys.argv[6])

        if not os.path.exists(TrainingFile):
            print("file not found!")
            quit()

        if not algorithm in ["average", "euclid", "cosine", "pearson"]:
            print("invalid algorithm")
            quit()

        weightedTotal = 0.0
        weights = 0.0

        if algorithm == "average":
            prediction = avgStat(TrainingFile, movie_id)

        movieDB = pd.read_table(filepath_or_buffer = TrainingFile, delimiter = '\t', names = ['user_ID', 'movies', 'ratings', 'times'])
        user_ID = movieDB['user_ID'].unique()
        mainUsrRating = {}
        similarUsrs = {}

        if movie_id >= 0 and movie_id <= 1682 and user_id >= 0 and user_id <= 943:
            for row in movieDB.iloc[0][1:-1]:
                arr = movieDB[movieDB['user_ID'] == user_id]
            for row in arr.iterrows():
                mainUsrRating[row[1].loc['movies']] = row[1].loc['ratings']
        else:
            print("movie_id/user_id is invalid")
            quit()

        if algorithm == "euclid":
            for user in user_ID:
                otherUsrRating = movieDB[movieDB['user_ID'] == user]
                distance = euclidDistance(mainUsrRating, otherUsrRating)
                if not distance == None:
                    similarUsrs[user] = distance
            prediction = euclidRatingStat(movieDB, movie_id, similarUsrs, k, weightedTotal, weights)

        if algorithm == "pearson":
            for user in user_ID:
                otherUsrRating = movieDB[movieDB['user_ID'] == user]
                correlation = pearsonCorrelation(mainUsrRating, otherUsrRating)
                if not correlation == None:
                    similarUsrs[user] = correlation
            prediction = pearsonRatingStat(movieDB, movie_id, similarUsrs, k, weightedTotal, weights)

        if algorithm == "cosine":
            for user in user_ID:
                otherUsrRating = movieDB[movieDB['user_ID'] == user]
                cosSim = cosineSimilarity(mainUsrRating, otherUsrRating)
                if not cosSim == None:
                    similarUsrs[user] = cosSim
            prediction = cosineRatingStat(movieDB, movie_id, similarUsrs, k, weightedTotal, weights)

        print("myrex.command    = predict")
        print("myrex.training   = " + TrainingFile)
        print("myrex.algorithm  = " + algorithm)
        print("myrex.k          = " + str(k))
        print("myrex.userID     = " + str(user_id))
        print("myrex.movieID    = " + str(movie_id))
        print("myrex.prediction = " + str(prediction))

    ############################################################################
    # evaluate
    ############################################################################
    if command == "evaluate" and len(sys.argv) == 6:
        TrainingFile = sys.argv[2]
        k = int(sys.argv[3])
        if (k < 0):
            print("Invalid 'k' value!")
            quit()
        algorithm = sys.argv[4]
        TestingFile = sys.argv[5]
        if not os.path.exists(TrainingFile) or not os.path.exists(TestingFile):
            print("file(s) not found!")
            quit()

        if not algorithm in ["average", "euclid", "cosine", "pearson"]:
            print("invalid algorithm")
            quit()

        weightedTotal = 0.0
        weights = 0.0
        RMSE = 0.0
        ratingCount = 0

        movieDB = pd.read_table(filepath_or_buffer = TestingFile, delimiter = '\t', names = ['user_ID', 'movies', 'ratings', 'times'])

        for row in movieDB.iterrows():
            user_id = row[1].loc['user_ID']
            movie_id = row[1].loc['movies']
            rating = row[1].loc['ratings']
            prediction = None

            if algorithm == "average":
                prediction = avgStat(TrainingFile, movie_id)

            movieDB2 = pd.read_table(filepath_or_buffer = TrainingFile, delimiter = '\t', names = ['user_ID', 'movies', 'ratings', 'times'])
            user_ID = movieDB['user_ID'].unique()
            mainUsrRating = {}
            similarUsrs = {}

            prediction = None

            for row in movieDB2.iloc[0][1:-1]:
                arr = movieDB2[movieDB2['user_ID'] == user_id]
            for row in arr.iterrows():
                mainUsrRating[row[1].loc['movies']] = row[1].loc['ratings']

            if movie_id < 0 or movie_id > 1682 or user_id < 0 or user_id > 943:
                print("movie_id/user_id is invalid")

            if algorithm == "euclid":
                weightedTotal = 0.0
                weights = 0.0

                for user in user_ID:
                    otherUsrRating = movieDB2[movieDB2['user_ID'] == user]
                    distance = euclidDistance(mainUsrRating, otherUsrRating)
                    if not distance == None:
                        similarUsrs[user] = distance
                prediction = euclidRatingStat(movieDB2, movie_id, similarUsrs, k, weightedTotal, weights)

            if algorithm == "pearson":
                weightedTotal = 0.0
                weights = 0.0
                for user in user_ID:
                    otherUsrRating = movieDB2[movieDB2['user_ID'] == user]
                    correlation = pearsonCorrelation(mainUsrRating, otherUsrRating)
                    if not correlation == None:
                        similarUsrs[user] = correlation
                prediction = pearsonRatingStat(movieDB2, movie_id, similarUsrs, k, weightedTotal, weights)

            if algorithm == "cosine":
                weightedTotal = 0.0
                weights = 0.0
                for user in user_ID:
                    otherUsrRating = movieDB2[movieDB2['user_ID'] == user]
                    cosSim = cosineSimilarity(mainUsrRating, otherUsrRating)
                    if not cosSim == None:
                        similarUsrs[user] = cosSim
                prediction = cosineRatingStat(movieDB2, movie_id, similarUsrs, k, weightedTotal, weights)

            if prediction != None:
                RMSE = RMSE + ((prediction - rating)**2)
                ratingCount = ratingCount + 1

        if ratingCount > 0:
            RMSE = math.sqrt(RMSE/ratingCount)
            print("myrex.command    = evaluate")
            print("myrex.training   = " + TrainingFile)
            print("myrex.testing    = " + TestingFile)
            print("myrex.algorithm  = " + algorithm)
            print("myrex.k          = " + str(k))
            print("myrex.RMSE       = " + str(RMSE))
        else:
            print("err, no predictions")
    else:
        print("Invalid number of args!")
else:
    print("Invalid number of args!")
