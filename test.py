import pandas as pd


pd.testing.assert_frame_equal(pd.read_csv('/workspaces/ChatbotWiz/data/processed/topics.csv'), pd.read_csv('/workspaces/ChatbotWiz/data/processed/topics_OLD.csv'))

