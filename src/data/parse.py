import hydra
import re

import pandas as pd

from urllib.parse import unquote_plus



def getSender(event):
    try:
        sender = re.search(r'^[\s\S]*{(.*?)}transType ', event).group(1)
    except AttributeError:
        sender = "SENDER NOT FOUND"
    
    return sender


def getMessage(event):
    try:
        msg = re.search(r'^[\s\S]*{(.*?)}fAct ', event).group(1)
    except AttributeError:
        msg = "MESSAGE NOT FOUND"
    
    return msg


def firstMessageUser(row):

    # Split all events from converation
    events = re.findall(r'{(.*?)}}', row['unquote_str'])
    # For each event, find sender
    for event in events:
        sender = getSender(event)
        if sender == 'CUSTOMER':
            return getMessage(event)
            


@hydra.main(config_path="../../conf", config_name="config")
def main(config):
    # Load data
    with open(config['raw_data_path'], 'r') as file:
        raw_str = file.read()

    # Decode 
    unquote_str =  unquote_plus(raw_str,
        encoding=config['encoding'], 
        errors='strict',
    )

    # Split by conversation
    conversations = unquote_str.split('\n')

    # Put in DataFrame format
    df_conv = pd.DataFrame(conversations, columns=['unquote_str'])

    # For each conversation, get first message sent by customer
    df_conv['first_msg_user'] = df_conv.apply(firstMessageUser, axis=1)

    # Clean
    df_conv = df_conv.dropna()
    
    print(df_conv)

    df_conv.to_csv(config['topic_modelling_folder']+config['first_message'])



if __name__ == "__main__":
    main()


