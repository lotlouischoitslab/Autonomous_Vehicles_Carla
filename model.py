import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 

from keras.models import Sequential 
from keras.optimizers import Adam 
from keras.callbacks import ModelCheckpoint 
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten 
from utils import INPUT_SHAPE, batch_generator 
 
import argparse 
import os 

np.random.seed(0)

def load_data(args):
    data_df = pd.read_csv(os.path.join(args.data_dir,'driving_log.csv'))

    X = data_df[['center','left','right']].values
    y = data_df['steering'].values 

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=args.test_size,random_state=0)

    return X_train,X_test,y_train,y_test