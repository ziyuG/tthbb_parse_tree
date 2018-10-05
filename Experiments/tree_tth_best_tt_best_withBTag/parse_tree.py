import sys, os
import copy
import time
#----------------------------
# fix random seed for reproducibility
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(1)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#---------------------------                                                    

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.models import load_model
import math as m
#Import my modules
sys.path.append(os.path.abspath('../../'))
from my_data import prepro
from my_data import prepare_ttbest_btag as m_prepare
from my_structures import model_nn_btag as lunch_model
from my_draw import train_monitor,draw_plots
from my_callback.sd_callback import roc_cb_earlyStop


"""
This is training macro of parse tree, with train on even events for the moment
"""

"""
More updates are needed:
#1. Check the correct way of monitoring training epoch.
2. Use a better callback function
"""
job_option = sys.argv[1]

#sig_file = '/home/guo/PycharmProjects/data/new_file.root'
sig_file = '/data1/home/ziyu.guo/data/ClassificationForTreeDNN_4j_180912.root'

bkg_file = sig_file
var_order = ["TTHReco_best_truthMatchPattern",
             "TTHReco_best_q1hadW_pT",     "TTHReco_best_q1hadW_eta",     "TTHReco_best_q1hadW_phi",     "TTHReco_best_q1hadW_E", "TTHReco_best_q1hadW_tagWeightBin",
             "TTHReco_best_q2hadW_pT",     "TTHReco_best_q2hadW_eta",     "TTHReco_best_q2hadW_phi",     "TTHReco_best_q2hadW_E", "TTHReco_best_q2hadW_tagWeightBin",
             "TTHReco_best_bhadTop_pT",    "TTHReco_best_bhadTop_eta",    "TTHReco_best_bhadTop_phi",    "TTHReco_best_bhadTop_E", "TTHReco_best_bhadTop_tagWeightBin",
             "lepton_pt",                  "lepton_eta",                  "lepton_phi",                  "lepton_E", "lepton_btag",
             "TTHReco_best_nulepTop_pT",   "TTHReco_best_nulepTop_eta",   "TTHReco_best_nulepTop_phi",   "TTHReco_best_nulepTop_E", "TTHReco_best_nulepTop_btag",
             "TTHReco_best_blepTop_pT",    "TTHReco_best_blepTop_eta",    "TTHReco_best_blepTop_phi",    "TTHReco_best_blepTop_E", "TTHReco_best_blepTop_tagWeightBin",
             "TTHReco_best_b1Higgsmv2_pT", "TTHReco_best_b1Higgsmv2_eta", "TTHReco_best_b1Higgsmv2_phi", "TTHReco_best_b1Higgsmv2_E", "TTHReco_best_bHiggs_tagWeightBin_1",
             "TTHReco_best_b2Higgsmv2_pT", "TTHReco_best_b2Higgsmv2_eta", "TTHReco_best_b2Higgsmv2_phi", "TTHReco_best_b2Higgsmv2_E", "TTHReco_best_bHiggs_tagWeightBin_2"]

#Obtained variables: X, Y, eventNumber, sample_weight, weight (5 in total) are default obtained vars. List the additional expected vars here.
var_obt = ["ClassifBDTOutput_inclusive_withBTag_new"]

cut_d = {'nBTags_85':'>= 4'}
sig_obt_dict, bkg_obt_dict = m_prepare.data_prepare(sig_file, bkg_file, var_order, var_obt, verbose=1, **cut_d)


m_prepare.match_filter('background')(sig_obt_dict)
m_prepare.match_filter('background')(bkg_obt_dict)


data_dict = m_prepare.merge_sig_bkg(sig_obt_dict, bkg_obt_dict, do_debug = False)

data_dict = m_prepare.lorentz_trans(data_dict)

# print data_dict['X']
# print data_dict['Y']
# print data_dict['weight']
# print data_dict['sample_weight']
# print data_dict['eventNumber']

##################### train, val, test splitting #####################################
eventNumber = data_dict['eventNumber']
        
X_e, X_o = m_prepare.even_odd_split(data_dict['X'], eventNumber)
Y_e, Y_o = m_prepare.even_odd_split(data_dict['Y'], eventNumber)
ClassifBDTOutput_inclusive_withBTag_new_e, ClassifBDTOutput_inclusive_withBTag_new_o = m_prepare.even_odd_split(data_dict['ClassifBDTOutput_inclusive_withBTag_new'], eventNumber)
weight_e, weight_o = m_prepare.even_odd_split(data_dict['weight'], eventNumber)
sample_weight_e, sample_weight_o = m_prepare.even_odd_split(data_dict['sample_weight'], eventNumber)
sample_weight_e = m_prepare.balance_class(sample_weight_e, Y_e)
sample_weight_o = m_prepare.balance_class(sample_weight_o, Y_o)

##################################
##### train_even and test_odd #####
val_split=0.2
X_e_learn, X_e_val = prepro.learn_val_split(X_e, val_split)
X_e_learn, X_o_app, X_e_val, l_mean, l_std = prepro.scale_norm(X_e_learn, X_o, X_validation=X_e_val)

Y_e_learn, Y_e_val = prepro.learn_val_split(Y_e, val_split)
sample_weight_e_learn, sample_weight_e_val = prepro.learn_val_split(sample_weight_e, val_split)
weight_e_learn, weight_e_val = prepro.learn_val_split(weight_e, val_split)
# print X_e_learn
# print Y_e_learn
# print sample_weight_e_learn
# print weight_e_learn
Y_o_app = np.copy(Y_o)
sample_weight_o_app = np.copy(sample_weight_o)
weight_o_app = np.copy(weight_o)


print("# of events for train: %d, val: %d, test: %d" % (len(Y_e_learn), len(Y_e_val), len(Y_o_app) ) )
if(job_option=="train" or job_option=="auto"):
    model_1 = lunch_model.m_model()
    model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model_1.summary()

    input_X = [X_e_learn[:, :5], X_e_learn[:, 5: 10], X_e_learn[:, 10: 15], X_e_learn[:, 15 :20], X_e_learn[:, 20: 25], X_e_learn[:, 25: 30], X_e_learn[:, 30: 35], X_e_learn[:, 35: 40]]
    input_Y = Y_e_learn

    val_X = [X_e_val[:, :5], X_e_val[:, 5: 10], X_e_val[:, 10: 15], X_e_val[:, 15 :20], X_e_val[:, 20: 25], X_e_val[:, 25: 30], X_e_val[:, 30: 35], X_e_val[:, 35: 40]]
    val_Y = Y_e_val

    callbacks_auc_1 = roc_cb_earlyStop(input_X, val_X, input_Y, val_Y, weight_e_learn, weight_e_val)
    callbacks= [# ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
                callbacks_auc_1]


    history_1 = model_1.fit(input_X, input_Y, epochs=50, batch_size=200, sample_weight=sample_weight_e_learn, callbacks=callbacks
    , validation_data=(val_X, val_Y, sample_weight_e_val))

    train_monitor.mon_training("model_even_odd", history_1, "loss")
    train_monitor.mon_training("model_even_odd", history_1, "acc")
    train_monitor.mon_auc("model_even_odd", callbacks_auc_1)

if(job_option=="prediction" or job_option=="auto" ):
    from my_prediction.auto_pred import get_best_model, save_roc_app
    from sklearn.metrics import roc_curve, roc_auc_score

    app_X = [X_o_app[:, :5], X_o_app[:, 5: 10], X_o_app[:, 10: 15], X_o_app[:, 15 :20], X_o_app[:, 20: 25], X_o_app[:, 25: 30], X_o_app[:, 30: 35], X_o_app[:, 35:40]]

    print ("Weight and arch saved separately:")
    new_model = lunch_model.m_model()
    new_model_weight, max_epoch, max_auc = get_best_model("Weight-*-*.h5")
    new_model.load_weights(new_model_weight)
    new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    p_test_odd = new_model.predict(app_X, batch_size=200, verbose=0)
    print roc_auc_score(Y_o_app, p_test_odd, sample_weight=weight_o_app)
    
    #----- Save ROC for model comparison -----
    fpr, tpr, threshold = roc_curve(Y_o_app, p_test_odd, sample_weight=weight_o_app)
    save_roc_app(fpr, tpr, threshold, [max_epoch, max_auc])
    # #----- Draw ROC, compared with BDT -----
    # ref_dict = {'BDT': ClassifBDTOutput_inclusive_withBTag_new_o}
    # draw_plots.drawROC_pred(Y_o_app, p_test_odd, weight_o_app, **ref_dict)
