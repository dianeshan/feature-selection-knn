import copy
import math
import numpy as np
import pandas as pd

def main():

    print("Welcome to Feature Selection with Nearest Neighbor! Get ready to get real cozy with your neighbors :)\n")
    fileName = input("Type in the name of the file to test: ")

    # get algorithm from input
    algorithm = input(
        "Type the number(1 or 2) of the algorithm you want to run.\n1) Forward Selection\n2) Backward Elimination\n")

    algo = int(algorithm)  # turn algorithm into an integer

    if algo == 1:
        print("\nForward Selection selected\n")
    elif algo == 2:
        print("\nBackward Elimination selected\n")

    # load in data here
    data = []

    with open(fileName) as f:
        for line in f:
            feats = line.split()
            data.append(feats)  # each instance is added to array

    features = len(data[0])
    num_features = features - 1
    instances = len(data)

    # initialize searched_features with all the features in it for the first LOO cross validation
    searched_features = set()
    for i in range(1, features):
        searched_features.add(i)

    # turn array into dataframe and then convert to numerics in order to do easier calculations later
    # https://stackoverflow.com/questions/34844711/convert-entire-pandas-dataframe-to-integers-in-pandas-0-17-0
    df = pd.DataFrame(data)
    data_df = df.apply(pd.to_numeric)

    print("This dataset has " + str(num_features) +
          " features (not including the class attribute), with " + str(instances) + " instances.\n")

    acc = loo_cross_validation(instances, data_df, features, searched_features)

    # how to format percentage nicely in python: https://www.adamsmith.haus/python/answers/how-to-format-a-number-as-a-percentage-in-python
    print('Running nearest neighbor with all ' + str(num_features) +
          ' features, using "leave-one-out" cross validation, I get an accuracy of ' + "{:.1%}".format(acc) + '\n')

    print('Beginning Search:\n')

    if algo == 1:
        return forward_selection(data_df, features, instances)
    elif algo == 2:
        return backward_elimination(data_df, features, instances, searched_features)


def loo_cross_validation(instances, data_df, features, searched_features):
    correctly_classified_count = 0

    # turn into an array in order to do calculations correctly
    data_arr = np.array(data_df)

    for j in range(1, features):
        if j not in searched_features:
            data_arr[:, j] = 0.0

    for i in range(instances):
        # initialize nearest neighbor distance to infinity so we can set first distance to it
        nearest_neighbor_distance = math.inf
        # initialize nearest neighbor location to infinity
        nearest_neighbor_location = math.inf

        # takes the features of the current instance we are on
        object_to_classify = data_arr[i][1:features]
        label_object_to_classify = data_arr[i][0]  # gets the label

        for k in range(instances):
            if k != i:
                distance = math.sqrt(
                    sum(pow(object_to_classify - data_arr[k][1:features], 2)))  # use euclidean distance formula to calculate distance

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data_arr[nearest_neighbor_location][0]

        if label_object_to_classify == nearest_neighbor_label:
            correctly_classified_count = correctly_classified_count + 1

    accuracy = correctly_classified_count / instances
    return accuracy

def forward_selection(data_df, features, instances):

    # making it a set to guarantee that there are no repeats of features in here
    searched_features = set()
    # making this a map so we can easily access the accuracy and the set of features
    best_feature_subsets = {}

    for i in range(1, features):
        max_acc = 0  # initialize maximum accuracy to 0
        best_feature = 0  # initialize best feature to 0

        for j in range(1, features):
            if j not in searched_features:

                # need to deepcopy otherwise not copied correctly and testing_features isn't updated
                # https://docs.python.org/3/library/copy.html
                testing_features = copy.deepcopy(searched_features)
                testing_features.add(j)

                acc = loo_cross_validation(
                    instances, data_df, features, testing_features)
                feat = j

                print("Using feature(s) " + str(testing_features) +
                      " accuracy is " + "{:.1%}".format(acc))

                # if the current accuracy is greater than max_acc then set max_acc equal
                # set final accuracy to acc and assign that feature to best_feature
                if acc > max_acc:
                    max_acc = acc
                    fin_acc = acc
                    best_feature = feat

        # add best feature into searched set
        searched_features.add(best_feature)
        # need to deepcopy otherwise not copied correctly and searched_features isn't updated
        searched_copy = copy.deepcopy(searched_features)
        # populating the map with maximum percentage and set of features from this iteration
        best_feature_subsets[fin_acc] = searched_copy

        print("Feature set " + str(searched_features) +
              " was best, accuracy is " + "{:.1%}".format(fin_acc) + "\n")

    best_subset_acc = max(best_feature_subsets.keys())
    print("YAY! Finished search!! The best feature subset is " + str(best_feature_subsets[best_subset_acc]
                                                                     ) + ", which has an accuracy of " + "{:.1%}".format(best_subset_acc) + "\n")


def backward_elimination(data_df, features, instances, searched_features):
    # making this a map so we can easily access the accuracy and the set of features
    best_feature_subsets = {}

    # run with all the features before going through loop
    acc = loo_cross_validation(
        instances, data_df, features, searched_features)
    print("Using feature(s) " + str(searched_features) +
          " accuracy is " + "{:.1%}".format(acc))
    print("Feature set " + str(searched_features) +
          " was best, accuracy is " + "{:.1%}".format(acc) + "\n")

    for i in range(1, features):
        max_acc = 0  # initialize maximum accuracy to 0
        best_feature = 0  # initialize best feature to 0

        for j in range(1, features):
            if j in searched_features:

                # need to deepcopy otherwise not copied correctly and testing_features isn't updated
                testing_features = copy.deepcopy(searched_features)
                testing_features.remove(j)

                acc = loo_cross_validation(
                    instances, data_df, features, testing_features)
                feat = j

                print("Using feature(s) " + str(testing_features) +
                      " accuracy is " + "{:.1%}".format(acc))

                # if the current accuracy is greater than max_acc then set max_acc equal
                # set final accuracy to acc and assign that feature to best_feature
                if acc > max_acc:
                    max_acc = acc
                    fin_acc = acc
                    best_feature = feat

        # add best feature into searched set
        searched_features.remove(best_feature)
        # need to deepcopy otherwise not copied correctly and searched_features isn't updated
        searched_copy = copy.deepcopy(searched_features)
        # populating the map with maximum percentage and set of features from this iteration
        best_feature_subsets[fin_acc] = searched_copy

        print("Feature set " + str(searched_features) +
              " was best, accuracy is " + "{:.1%}".format(fin_acc) + "\n")

    best_subset_acc = max(best_feature_subsets.keys())
    print("YAY! Finished search!! The best feature subset is " + str(best_feature_subsets[best_subset_acc]
                                                                     ) + ", which has an accuracy of " + "{:.1%}".format(best_subset_acc) + "\n")


if __name__ == "__main__":
    main()
