import hydra

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
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

    # Basic model training
    text_clf.fit(X_train, y_train)

    # Save model
    dump(text_clf, config['models_folder']+config['clf_base']) 



if __name__ == "__main__":
    main()
