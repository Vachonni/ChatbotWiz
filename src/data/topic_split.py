import hydra

import pandas as pd 

from sklearn.model_selection import train_test_split



@hydra.main(config_path="../../conf", config_name="config")
def main(config):  
   
    # Get first message and topic
    df_conv = pd.read_csv(config['topic_modelling_folder']+config['first_message_topic'])

    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(
        df_conv['first_msg_user'], 
        df_conv['topics_id'], 
        test_size=0.2, 
        random_state=config['seed'], 
        stratify=df_conv['topics_id'],
    )

    # Save data splits
    X_train.to_csv(config['topic_clf_folder']+config['X_train'], index=False)
    y_train.to_csv(config['topic_clf_folder']+config['y_train'], index=False)
    X_test.to_csv(config['topic_clf_folder']+config['X_test'], index=False)
    y_test.to_csv(config['topic_clf_folder']+config['y_test'], index=False)



if __name__ == "__main__":
    main()