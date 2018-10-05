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

#from rootpy.vector import LorentzVector

############################################
################ Define model ###############
def kins_trans(x):
    e = x[:, -1]
    pt = (x[:, 0]**2 + x[:, 1]**2)**0.5
    p = (x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5
    #m = (e**2 - p**2)**0.5
    phi = tf.atan(x[:, 1]/x[:, 0])
    theta = tf.atan(pt / x[:, 2])
    eta = -tf.log(tf.tan(theta/2.))

    kins = tf.stack([p, eta, theta, phi, e, pt], 1)
    #print kins
    return kins

def m_model():
    d=50
    #Map o to u
    shared_o2u = Dense(d, activation='relu')

    #Wqq
    input_q1w = Input(shape=(4,), name='input_lqw')
    input_q2w = Input(shape=(4,), name='input_rqw')
    k_q1w = (input_q1w)
    u_q1w = shared_o2u(k_q1w)##
    k_q2w = (input_q2w)
    u_q2w = shared_o2u(k_q2w)##
    h_q1w = u_q1w
    h_q2w = u_q2w
    
    o_hw = keras.layers.add([input_q1w, input_q2w])
    k_hw = (o_hw)
    u_hw = shared_o2u(k_hw)##

    h_hw_merge = keras.layers.concatenate([h_q1w, h_q2w, u_hw])
    h_hw = Dense(d, activation="relu")(h_hw_merge)#

    #htop
    input_bhtop = Input(shape=(4,), name='input_bhtop')
    k_bhtop = (input_bhtop)
    u_bhtop = shared_o2u(k_bhtop)##
    h_bhtop = u_bhtop
    
    o_htop = keras.layers.add([o_hw, input_bhtop])
    k_htop = (o_htop)
    u_htop = shared_o2u(k_htop)##
    
    h_htop_merge = keras.layers.concatenate([h_hw, h_bhtop, u_htop])
    h_htop = Dense(d, activation="relu")(h_htop_merge)#

    #wlv
    input_llep = Input(shape=(4,), name='input_llep')
    input_rneu = Input(shape=(4,), name='input_rneu')
    k_llep = (input_llep)
    u_llep = shared_o2u(k_llep)##
    k_rneu = (input_rneu)
    u_rneu = shared_o2u(k_rneu)##
    h_llep = u_llep  
    h_renu = u_rneu
    
    o_lw = keras.layers.add([input_llep, input_rneu])
    k_lw = (o_lw)
    u_lw = shared_o2u(k_lw)##

    h_lw_merge = keras.layers.concatenate([h_llep, h_renu, u_lw])
    h_lw = Dense(d, activation="relu")(h_lw_merge)#

    #ltop
    input_bltop = Input(shape=(4,), name='input_bltop')
    k_bltop = (input_bltop)
    u_bltop = shared_o2u(k_bltop)##
    h_bltop = u_bltop
    o_ltop = keras.layers.add([o_lw, input_bltop])
    k_ltop = (o_ltop)
    u_ltop = shared_o2u(k_ltop)##

    h_ltop_merge = keras.layers.concatenate([h_lw, h_bltop, u_ltop])
    h_ltop = Dense(d, activation="relu")(h_ltop_merge)#

    #ttbar
    o_tt = keras.layers.add([o_ltop, o_htop])
    k_tt = (o_tt)
    u_tt = shared_o2u(k_tt)##

    h_tt_merge = keras.layers.concatenate([h_ltop, h_htop, u_tt])
    h_tt = Dense(d, activation="relu")(h_tt_merge)#

    #Hbb
    input_b1h = Input(shape=(4,), name='input_b1h')
    input_b2h = Input(shape=(4,), name='input_b2h')
    k_b1h = (input_b1h)
    k_b2h = (input_b2h)
    u_b1h = shared_o2u(k_b1h)##
    u_b2h = shared_o2u(k_b2h)##
    h_b1h = u_b1h
    h_b2h = u_b2h
    
    o_higgs = keras.layers.add([input_b1h, input_b2h])
    k_higgs = (o_higgs)
    u_higgs = shared_o2u(k_higgs)##
    
    h_higgs_merge = keras.layers.concatenate([h_b1h, h_b2h, u_higgs])
    h_higgs = Dense(d, activation="relu")(h_higgs_merge)#

    #Collision root
    o_coll = keras.layers.add([o_tt, o_higgs])
    k_coll = (o_coll)
    u_coll = shared_o2u(k_coll)##

    h_coll_merge = keras.layers.concatenate([h_tt, h_higgs, u_coll])
    h_coll = Dense(d, activation="relu")(h_coll_merge)#

    #Classifier
    x = Dense(30, activation="relu")(h_coll)#
    x = Dense(30, activation="relu")(x)#
    m_output = Dense(1, activation="sigmoid")(x)#

    model = Model(input=[input_q1w, input_q2w, input_bhtop, input_llep, input_rneu, input_bltop, input_b1h, input_b2h], output=m_output)

    
    #print(model.summary())
    #plot(model, to_file='model_rnn1_plot.png', show_shapes=True, show_layer_names=True)
    
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.load_weights('weights-Fold-1-improvement-00-0.156.hdf')
    return model

#model = m_model()
