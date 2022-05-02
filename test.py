import pandas as pd


pd.testing.assert_frame_equal(pd.read_csv('/workspaces/ChatbotWiz/data/processed/topic_modelling/first_message_topic.csv'), pd.read_csv('/workspaces/ChatbotWiz/data/processed/topics.csv'))

