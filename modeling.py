from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
import os
from lstm_neural_nets import lstm_model_with_dropout_architecture, lstm_model_without_dropout_architecture
from config import config

MODEL_PATH = config['MODEL_PATH']

def scores(confusion_matrix,model_name):
    """
    This function is used to generate scores when confusion matrix is given

    :param confusion_matrix: confusion matrix
    :param model_name: The model name of which confusion matrix is generated
    :return: None
    """
    # extract the scores from the confusion matrix
    tn, fp, fn, tp = confusion_matrix.ravel()
    print("="*10)
    print(model_name)

    # calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # print the results
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
    print("="*10)


def model(df):
    """
    This function controls our modeling flow
    we have 6 machine learning models along with couple of variants of LSTM to model our
    inferences.
    :param df: input dataframe
    :return:
    """
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    cv = CountVectorizer(max_features = 5000)
    df = df.iloc[:1000]

    X_title = cv.fit_transform(df.clean_joined_title).toarray()
    y_target = df.target.values

    X_train, X_test, y_train, y_test = train_test_split(X_title, y_target, test_size = 0.20,
                                                        random_state = 0)

    # scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # define the classifiers
    classifier_gaussian_nb = GaussianNB()
    classifier_bernoulli_nb = BernoulliNB()
    classifier_logistic_reg = LogisticRegression(max_iter=10000)
    classifier_decision_tree = DecisionTreeClassifier()
    classifier_random_forest = RandomForestClassifier()
    classifier_knn = KNeighborsClassifier()

    # define the parameter grids for hyperparameter tuning
    param_grid_logistic_reg = {'C': [0.1, 1, 10, 100]}
    param_grid_decision_tree = {'max_depth': [2, 4, 6, 8, 10]}
    param_grid_random_forest = {'n_estimators': [10, 50, 100, 500]}
    param_grid_knn = {'n_neighbors': [3, 5, 7, 9]}

    # define a dictionary of classifiers and their parameter grids
    classifiers = {
        'Gaussian Naive Bayes': (classifier_gaussian_nb, {}),
        'Bernoulli Naive Bayes': (classifier_bernoulli_nb, {}),
        'Logistic Regression': (classifier_logistic_reg, param_grid_logistic_reg),
        'Decision Tree': (classifier_decision_tree, param_grid_decision_tree),
        'Random Forest': (classifier_random_forest, param_grid_random_forest),
        'K-Nearest Neighbors': (classifier_knn, param_grid_knn),
    }

    if MODEL_PATH != 'saved':
        for name, (classifier,dictionary) in classifiers.items():
            if type(classifier) == LogisticRegression:
                classifier.fit(X_train_scaled, y_train)
            else:
                classifier.fit(X_train, y_train)
            joblib.dump(classifier, f'{MODEL_PATH}/model_{name}.pkl')

    # make predictions and calculate confusion matrices
    for name in classifiers:
        classifier = joblib.load(f'{MODEL_PATH}/model_{name}.pkl')
        if type(classifier) == LogisticRegression:
            y_pred = classifier.predict(X_test_scaled)
        else:
            y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        scores(cm, name)


    """We will perform grid search for Logistic regression and Decision Tree"""
    # define a dictionary of classifiers and their parameter grids
    classifiers = {
        'Logistic Regression': (classifier_logistic_reg, param_grid_logistic_reg),
        'Decision Tree': (classifier_decision_tree, param_grid_decision_tree)
    }
    if MODEL_PATH != 'saved':
        for name, (classifier, param_grid) in classifiers.items():
            if type(classifier) == LogisticRegression:
                grid_search = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
            else:
                grid_search = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1)
                grid_search.fit(X_train, y_train)
            best_classifier = grid_search.best_estimator_
            joblib.dump(best_classifier, f'{MODEL_PATH}/grid_search_model_{name}.pkl')

    # make predictions and calculate confusion matrices
    for name in classifiers:
        classifier = joblib.load(f'{MODEL_PATH}/grid_search_model_{name}.pkl')
        if type(classifier) == LogisticRegression:
            y_pred = classifier.predict(X_test_scaled)
        else:
            y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        scores(cm, name)


    train_size = int(0.8 * len(df))
    train_texts = df['clean_joined_text'][:train_size]
    train_labels = df['target'][:train_size]
    test_texts = df['clean_joined_text'][train_size:]
    test_labels = df['target'][train_size:]
    lstm_model_with_dropout_architecture(train_texts,test_texts,train_labels,test_labels)
    lstm_model_without_dropout_architecture(train_texts,test_texts,train_labels,test_labels)

