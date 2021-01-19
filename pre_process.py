
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import pandas as pd 
import torch
from config import c_file,b_file,c_index,b_index,process_path
from utils import get_logger 

if __name__=='__main__':
    logger = get_logger()

    logger.info('读取click..')
    df = pd.read_csv(c_file, header=None, names=c_index,low_memory=False)
    logger.info(df.head(20))

    logger.info('读取buy..')
    buy_df = pd.read_csv(b_file, header=None, names=b_index)
    logger.info(buy_df.head(20))

    item_encoder = LabelEncoder()
    df['item_id'] = item_encoder.fit_transform(df.item_id)
    logger.info(df.head())
    #randomly sample 
    sample_session_id = np.random.choice(df.session_id.unique(), 1000000, replace=False)
    df = df.loc[df.session_id.isin(sample_session_id)]
    logger.info(df.nunique())

    df['label'] = df.session_id.isin(buy_df.session_id)
    logger.info(df.head())

    with open(process_path,'wb') as f:
        torch.save(df,f)

    num_embeddings = df.item_id.max() + 1
    logger.info('num_embedding:' + str(num_embeddings))

    logger.info('预处理完毕')