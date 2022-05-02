import hydra
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from joblib import dump



@hydra.main(config_path="../../conf", config_name="config")
def main(config):

    # Get data, first message and topic
    df_conv = pd.read_csv(config['first_message_topic_path'])

    # Split data 
    X_train, _, y_train, _ = train_test_split(
        df_conv['first_msg_user'], 
        df_conv['topics_id'], 
        test_size=0.2, 
        random_state=config['seed'], 
        stratify=df_conv['topics_id'],
    )

    # Pipeline preprocess text data and create model
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',  
                            alpha=1e-2, random_state=config['seed'],
                            max_iter=5, tol=None)),
    ])

    # Create cross validation splits, stratified
    cv = StratifiedShuffleSplit(n_splits=5, random_state=config['seed'])

    # HP Search model
    params = dict(config['parameters'])   # Need to be dict for HP search
    gs_clf = GridSearchCV(text_clf, params, cv=cv, n_jobs=-1, scoring=config['score_metric'])
    gs_clf = gs_clf.fit(X_train, y_train)
    print(f'\n**HP CLF SCORE: {gs_clf.best_score_} and Best HP found:')
    for param_name in sorted(params.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    print(pd.DataFrame(gs_clf.cv_results_)
    )

    # Save model
    dump(gs_clf, config['models_folder']+config['clf_HP_search']) 



if __name__ == "__main__":
    main()
