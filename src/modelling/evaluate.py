import hydra
import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn import metrics
from joblib import load
from transformers import pipeline


logger = logging.getLogger(__name__)


def evaluate_HGF_zero_shot(X_test, y_test):

    # Topics and id correspondances
    labels_str = {'code password log new': 0, 
                'printer print scan attached': 1, 
                'ticket follow': 2}

    # Get zero-shot-classfier
    classifier = pipeline('zero-shot-classification')
    candidate_labels = list(labels_str.keys())

    # Predict topics
    pipeline_pred_complete = classifier(list(X_test.values), candidate_labels, hypothesis_template="This is probably a conversation on the topic of {}")
    # Convert to id for scoring
    pipeline_pred_id = [labels_str[pred['labels'][0]] for pred in pipeline_pred_complete]
    logger.info(f'\n{metrics.classification_report(y_test, pipeline_pred_id)}')
    logger.info(f'Confusion matrix:\n{metrics.confusion_matrix(y_test, pipeline_pred_id)}')


def evaluation(model, metric, cv, X_train, X_test, y_train, y_test):

    # Traditional 
    predicted = model.predict(X_test)
    logger.debug(model.predict(X_test))
    logger.debug(X_test)
    logger.info(f'\n{metrics.classification_report(y_test, predicted)}')
    logger.info(f'Confusion matrix:\n{metrics.confusion_matrix(y_test, predicted)}')
    # dict_metrics = metrics.classification_report(y_test, predicted, output_dict=True)

    # Create cross validation splits, stratified
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
    logger.info(f"CV SCORES: {scores}, mean {scores.mean()} and std {scores.std()}")

    # return scores.mean(), scores.std(), dict_metrics['accuracy'], dict_metrics['weighted avg']['f1-score']



@hydra.main(config_path="../../conf", config_name="config")
def main(config):

    # Get data, first message and topic
    df_conv = pd.read_csv(config['first_message_topic_path'])

    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(
        df_conv['first_msg_user'], 
        df_conv['topics_id'], 
        test_size=0.2, 
        random_state=config['seed'], 
        stratify=df_conv['topics_id'],
    )

    # Create cross validation splits 
    cv = StratifiedShuffleSplit(n_splits=5, random_state=config['seed'])

    # Load models and evaluate them
    clf_base = load(config['models_folder']+config['clf_base']) 
    logger.info('-----Evaluating cfl_base')
    evaluation(clf_base, config['score_metric'], cv, X_train, X_test, y_train, y_test)

    clf_HP_search = load(config['models_folder']+config['clf_HP_search']) 
    logger.info('-----Evaluating cfl_base_HP_search')
    evaluation(clf_HP_search, config['score_metric'], cv, X_train, X_test, y_train, y_test)

    # Evaluate HugginFace zero-shot
    evaluate_HGF_zero_shot(X_test, y_test)



if __name__ == "__main__":
    main()