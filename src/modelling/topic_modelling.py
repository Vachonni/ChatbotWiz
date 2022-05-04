import hydra
import logging 

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic


logger = logging.getLogger(__name__)

def getTopicName(df_topic_info, topic_id):

    return df_topic_info.loc[df_topic_info['Topic'] == topic_id]['Name'].item()


@hydra.main(config_path="../../conf", config_name="config")
def main(config):

    # Instantiate topic model
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    topic_model = BERTopic(language="english", 
        min_topic_size=3,                # higher -> less topics, default=10.
        embedding_model="all-MiniLM-L6-v2", 
        vectorizer_model=vectorizer_model
    )

    # Get data
    df_conv = pd.read_csv(config['first_message_path'])

    # Train model and get topics id until 3 topics found
    n_topics = 0
    while n_topics != 3:
        df_conv['topics_id'], _ = topic_model.fit_transform(df_conv['first_msg_user'])
        n_topics = df_conv['topics_id'].nunique()
        logger.info(f'Number of topics found: {n_topics}')

    # Get topics info
    df_topic_info = topic_model.get_topic_info()

    # Include topic name 
    df_conv['topics_name'] = df_conv['topics_id'].apply(lambda x: getTopicName(df_topic_info, x))

    # Save 
    df_conv.to_csv(config['first_message_topic_path'], index=False)



if __name__ == "__main__":
    main()