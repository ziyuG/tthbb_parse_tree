from keras import backend as K
import tensorflow as tf
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from sklearn.ensemble import BaggingClassifier
from keras.wrappers.scikit_learn import KerasClassifier
# Create first network with Keras
from keras.layers import Dense, Reshape, Activation, Dropout, LSTM, Input
import keras
from keras.optimizers import RMSprop, Adamax, Adagrad, SGD
from sklearn.metrics import roc_auc_score, roc_curve
import math as m

def m_model(n_feature):
    input_x = Input(shape=(n_feature,), name='input_features')
    x = Dense(32, init='uniform')(input_x)
    x = Dense(300, init='glorot_uniform', activation="relu")(x)
    x = Dropout(0.129)(x)
    x = Dense(200, init='lecun_normal', activation="sigmoid")(x)
    x = Dropout(0.008)(x)
    x = Dense(200, init='uniform', activation="relu")(x)
    x = Dropout(0.161)(x)
    x = Dense(200, init='glorot_uniform', activation="relu")(x)
    x = Dropout(0.001)(x)
    x = Dense(400, init='uniform', activation="sigmoid")(x)
    x = Dropout(0.121)(x)
    output_x = Dense(1, init='glorot_normal', activation="sigmoid")(x)
    
    model = Model(input=input_x, output=output_x)

    
    #print(model.summary())
    #plot(model, to_file='model_rnn1_plot.png', show_shapes=True, show_layer_names=True)
    
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.load_weights('weights-Fold-1-improvement-00-0.156.hdf')
    return model

#model = m_model()