import hydra
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV



def evaluation(model, metric, cv, X_train, X_test, y_train, y_test):

    # Traditional 
    predicted = model.predict(X_test)
    accuracy = np.mean(predicted == y_test)
    print("**BASIC:")
    print(accuracy)
    print(model.predict(X_test))
    print(X_test)
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))

    # Cross va
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
    print(f"\n**CV SCORES BASIC: {scores}, mean {scores.mean()} and std {scores.std()} \n")



@hydra.main(config_path="../../conf", config_name="config")
def main(config):

    # Pipeline preprocess text data and create model
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',  
                            alpha=1e-2, random_state=config['seed'],
                            max_iter=5, tol=None)),
    ])

    # Create cross validation splits 
    cv = StratifiedShuffleSplit(n_splits=5, random_state=config['seed'])

    # Basic model
    text_clf.fit(X_train, y_train)


    text_clf2 = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',  
                            alpha=1e-2, random_state=config['seed'],
                            max_iter=5, tol=None)),
    ])

    # HP Search model
    params = dict(config['parameters'])   # Need to be dict for HP search
    gs_clf = GridSearchCV(text_clf2, params, cv=cv, n_jobs=-1, scoring=config['score_metric'])
    gs_clf = gs_clf.fit(X_train, y_train)
    print(f'\n**HP CLF SCORE: {gs_clf.best_score_} and Best HP found:')
    for param_name in sorted(params.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    print(pd.DataFrame(gs_clf.cv_results_)
    )
    evaluation(text_clf, config['score_metric'], cv, X_train, X_test, y_train, y_test)
    evaluation(gs_clf, config['score_metric'], cv, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
