import pickle
import pandas as pd
import pprint
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.spatial import distance

"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""
import joblib
import cvxopt
import copy


# def linear_kernel(x1, x2, Q=1):
#     output =  np.matmul(x1,x2.T)
#     print(f"TEST:{output.shape}")
#     return output
#
#
# def polynomial_kernel(x, y, p=3):
#     return (1 + np.dot(x, y)) ** p
#
#
# def gaussian_kernel(x, y, sigma=5.0):
#     return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))
#
#
# class SVM:
#     def __init__(self, kernel_function, C, Q, k):
#         self.kernel_function = kernel_function
#         self.C = C
#         self.Q = Q
#         self.k = k
#         self.classes = 100
#         self.clfs = []
#     def fit(self, X, y):
#         print(f"X:{X.shape}")
#         if self.classes > 1:
#             return self.multi_fit(X, y)
#         if set(np.unique(y)) == {0, 1}: y[y == 0] = -1
#         self.y = y.reshape(-1, 1).astype(np.double)
#         print(type(y))
#         print(y.size)
#         print(type(self.y))
#         self.X = X
#         points_size = X.shape[0]
#         self.K = self.kernel_function(X, X, self.Q).astype(np.double)
#         print(self.K.size)
#         P = cvxopt.matrix(self.y @ self.y.T * self.K)
#         print(P.size)
#         q = cvxopt.matrix(-np.ones((points_size, 1)))
#         print(q.size)
#         A = cvxopt.matrix(self.y.T)
#         b = cvxopt.matrix(np.zeros(1))
#         G = cvxopt.matrix(np.vstack((-np.identity(points_size),
#                                      np.identity(points_size))))
#         h = cvxopt.matrix(np.vstack((np.zeros((points_size, 1)),
#                                      np.ones((points_size, 1)) * self.C)))
#         sol = cvxopt.solvers.qp(P, q, G, h, A, b)
#         self.sol = np.array(sol["x"])
#         print(self.sol)
#         self.is_sv = ((self.sol - 1e-30 > 0) & (self.sol <= self.C)).squeeze()
#         print(self.is_sv)
#         self.margin_sv = np.argmax((0 < self.sol - 1e-30) & (self.sol < self.C - 1e-3))
#     def multi_fit(self, X, y):
#         self.classes = len(np.unique(y))
#         for i in range(self.classes):
#             temp_X, temp_y = X, y
#             temp_y[temp_y != i], temp_y[temp_y == i] = -1, +1
#             clf = SVM(kernel_function=self.kernel_function, C=self.C, k=self.Q, Q=self.k)
#             print(f"THERE={self.classes}")
#             self.fit(temp_X, temp_y)
#             self.clfs.append(clf)
#     def predict(self, X_pred):
#         if self.classes > 1:
#             return self.multi_predict(X_pred)
#         x_temp, y_temp = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
#         αs, y, X = self.sol[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]
#         b = y_temp - np.sum(αs * y * self.kernel_function(X, x_temp, self.classes), axis=0)
#         score = np.sum(αs * y * self.kernel_function(X, X_pred, self.classes), axis=0) + b
#         return np.sign(score).astype(int), score
#
#     def multi_predict(self, X):
#         N = X.shape[0]
#         preds = np.zeros((N, self.classes))
#         for i, clf in enumerate(self.clfs):
#             _, preds[:, i] = clf.predict(X, self.classes)
#         return np.argmax(preds, axis=1), np.max(preds, axis=1)
#
#     def evaluate(self, X, y):
#         output, _ = self.predict(X)
#         accuracy = accuracy_score(output)
#         return accuracy
#
#
def combine_pickle_files(filelist, output_file):
    with open(filelist[0], 'rb') as plik_1:
        dane_1 = pickle.load(plik_1)

    # Odczytaj dane z drugiego pliku Pickle
    with open(filelist[1], 'rb') as plik_2:
        dane_2 = pickle.load(plik_2)
    data_1 = dane_1["data"]
    data_2 = dane_2["data"]
    combined_pickle = {
        "bounds": dane_1["bounds"],
        "block_size": dane_1["block_size"],
        "data": data_1 + data_2
    }
    with open(output_file, 'wb') as handle:
        pickle.dump(combined_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_model(filename, model):
    joblib.dump(model, filename)


def split_dataframes(df, columns_x, columns_y):
    X_train, X_test, y_train, y_test = train_test_split(df[columns_x], df.filter(like=columns_y), test_size=0.2)
    return X_train, X_test, y_train, y_test


def train_model(df, columns_x, columns_y):
    X_train, X_test, y_train, y_test = split_dataframes(df, columns_x, columns_y)
    svm_classifier = SVC(kernel='linear', decision_function_shape='ovr')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return svm_classifier


def check_directions(head_x, head_y, element, block_size, directions):
    if head_y == element[1] + block_size:
        directions[0] = 1
    elif head_x == element[0] - block_size:
        directions[1] = 1
    elif head_y == element[1] - block_size:
        directions[2] = 1
    elif head_x == element[0] + block_size:
        directions[3] = 1
    return directions


def check_food_directions(head_x, head_y, food):
    food_dir = [food[1] < head_y, food[0] > head_x, food[1] > head_y, food[0] < head_x]
    return food_dir


def get_binaries(snake_body, bounds, block_size, food):
    head_x = snake_body[-1][0]
    head_y = snake_body[-1][1]
    obstacles = [0, 0, 0, 0]
    for element in snake_body:
        obstacles = check_directions(head_x, head_y, element, block_size, obstacles)
    binaries_obstacles = check_directions(head_x, head_y, bounds, block_size, obstacles)
    binaries_food = check_food_directions(head_x, head_y, food)
    return binaries_obstacles + binaries_food


def game_state_to_data_sample(game_state: dict, input_columns, output_columns):
    df = pd.DataFrame(game_state["data"])
    dummies_moves = df[1].rename(output_columns)
    bounds = game_state["bounds"]
    block_size = game_state["block_size"]
    new_df = pd.DataFrame()
    new_df[input_columns] = pd.DataFrame(
        df[0].apply(lambda row: get_binaries(row['snake_body'], bounds, block_size, row["food"])).tolist(),
        index=df.index)
    snake_df = pd.concat([new_df, dummies_moves], axis=1)
    snake_df[output_columns] = snake_df[output_columns].apply(lambda x: x.value)
    return snake_df


def linear_kernel(x1, x2, c=0):
    return x1 @ x2.T


def rbf(x1, x2, gamma=10):
    x1 = x1.astype(np.double)
    x2 = x2.astype(np.double)
    return np.exp(-gamma * distance.cdist(x1, x2))


class SVM:

    def __init__(self, kernel_function, C=0.1, k=4):
        self.kernel = kernel_function
        self.C = C
        self.k = k
        self.X, y = None, None
        self.αs = None
        self.multiclass = False
        self.clfs = []

    def fit(self, X, y, eval_train=False):
        if len(np.unique(y)) > 2:
            self.multiclass = True
            return self.multi_fit(X, y, eval_train)
        if set(np.unique(y)) == {0, 1}: y[y == 0] = -1
        self.y = y.reshape(-1, 1).astype(np.double)
        self.X = X
        N = X.shape[0]
        self.K = self.kernel(X, X, self.k).astype(np.double)

        # For 1/2 x^T P x + q^T x
        P = cvxopt.matrix(self.y @ self.y.T * self.K)
        q = cvxopt.matrix(-np.ones((N, 1)))

        # For Ax = b
        A = cvxopt.matrix(self.y.T)
        b = cvxopt.matrix(np.zeros(1))

        # For Gx <= h
        G = cvxopt.matrix(np.vstack((-np.identity(N),
                                     np.identity(N))))
        h = cvxopt.matrix(np.vstack((np.zeros((N, 1)),
                                     np.ones((N, 1)) * self.C)))

        # Solve
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.αs = np.array(sol["x"])

        # Maps into support vectors
        self.is_sv = ((self.αs > 1e-3) & (self.αs <= self.C)).squeeze()
        self.margin_sv = np.argmax((1e-3 < self.αs) & (self.αs < self.C - 1e-3))

        if eval_train:
            print(f"Finished training with accuracy {self.evaluate(X, y)}")

    def multi_fit(self, X, y, eval_train=False):
        self.k = len(np.unique(y))  # number of classes
        for i in range(self.k):
            Xs, Ys = X, copy.copy(y)
            # change the labels to -1 and 1
            Ys[Ys != i], Ys[Ys == i] = -1, +1
            # fit the classifier
            clf = SVM(kernel_function=self.kernel, C=self.C, k=self.k)
            clf.fit(Xs, Ys)
            # save the classifier
            self.clfs.append(clf)
        if eval_train:
            print(f"Finished training with accuracy {self.evaluate(X, y)}")

    def predict(self, X_t):
        """

        Args:
            X_t:

        Returns:

        """
        if self.multiclass: return self.multi_predict(X_t)
        xₛ, yₛ = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
        αs, y, X = self.αs[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]

        b = yₛ - np.sum(αs * y * self.kernel(X, xₛ, self.k), axis=0)
        score = np.sum(αs * y * self.kernel(X, X_t, self.k), axis=0) + b
        return np.sign(score).astype(int), score

    def multi_predict(self, X):
        # get the predictions from all classifiers
        preds = np.zeros((X.shape[0], self.k))
        for i, clf in enumerate(self.clfs):
            _, preds[:, i] = clf.predict(X)

        # get the argmax and the corresponding score
        return np.argmax(preds, axis=1)

    def evaluate(self, X, y):
        outputs = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)

    from sklearn.datasets import make_classification
    import numpy as np


# Test SVM
if __name__ == "__main__":
    # """ Example of how to read a pickled file, feel free to remove this""
    # combine_pickle_files(["snakerun5.pickle","snakerun4.pickle"], "output2.pickle")
    with open(f"snakerun5.pickle", 'rb') as f:
        data_file = pickle.load(f)
    pprint.pprint(data_file)
    output_columns = "Next move"
    input_columns = input = ['OBSTACLE_UP', 'OBSTACLE_RIGHT', 'OBSTACLE_DOWN', 'OBSTACLE_LEFT', 'FOOD_UP', 'FOOD_RIGHT',
                             'FOOD_DOWN',
                             'FOOD_LEFT']
    snake_dataframe = game_state_to_data_sample(game_state=data_file, input_columns=input_columns,
                                                output_columns=output_columns)
    model = train_model(snake_dataframe, input_columns, output_columns)
    save_model("michalxx.sav", model)
    print(snake_dataframe)
    test = snake_dataframe.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(test[:, :-1], test[:, -1], test_size=0.1)
    svm_model = SVM(kernel_function=rbf, k=4)
    svm_model.fit(X_train, y_train, eval_train=True)

    y_pred = svm_model.predict(X_test)
    print(y_pred)
    print(f"Accuracy: {np.sum(y_test == y_pred) / y_test.shape[0]}")
    test = pd.DataFrame()
    test["WER"] = y_test
    test["ACTUAL"] = y_pred
    test.to_csv("test.csv", index=False)
    # filename = "tescik.pkl"
    # with open(filename, 'wb') as file:
    #     pickle.dump(svm_model, file)
    joblib.dump(svm_model, "test.sav")
    # clf = OneVsRestClassifier(SVC(kernel='linear', C=1, gamma=4)).fit(X, y)
    # y_pred = clf.predict(X)
    # svm = SVM(linear_kernel,2,2,1)
    # svm.fit(X_train, y_train)
    # y_pred,score = svm.predict(X_test)
    # print(y_pred)
    # print(score)
    # save_model(filename="test.sav", model=model)
