import hydra

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic




def getTopicName(df_topic_info, topic_id):

    return df_topic_info.loc[df_topic_info['Topic'] == topic_id]['Name'].item()


@hydra.main(config_path="../../conf", config_name="config")
def main(config):

    # Instantiate topic model
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    topic_model = BERTopic(language="english", 
        min_topic_size=3, 
        embedding_model="all-MiniLM-L6-v2", 
        vectorizer_model=vectorizer_model
    )

    # Get data
    df_conv = pd.read_csv(config['conv_path'])

    # Train model and get topics id
    df_conv['topics_id'], _ = topic_model.fit_transform(df_conv['first_msg_user'])

    # Get topics info
    df_topic_info = topic_model.get_topic_info()

    # Include topic name 
    df_conv['topics_name'] = df_conv['topics_id'].apply(lambda x: getTopicName(df_topic_info, x))

    # Save 
    df_conv.to_csv(config['topics_path'], index=False)



if __name__ == "__main__":
    main()