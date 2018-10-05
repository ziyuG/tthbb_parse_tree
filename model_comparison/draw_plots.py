#from __future__ import unicode_literals
import sys
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
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from sklearn.ensemble import BaggingClassifier
from keras.wrappers.scikit_learn import KerasClassifier
# Create first network with Keras
from keras.layers import Dense, Reshape, Activation, Dropout, LSTM, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import RMSprop, Adamax, Adagrad, SGD
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_auc_score, roc_curve, auc
import csv
#import pandas
#from keras.regularizers import l1, activity_l1, l1l2
import matplotlib.pyplot as plt
#from keras.utils.visualize_util import plot
import uproot
import pickle
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib import colors
import glob

#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True

#np.seterr(divide='ignore', invalid='ignore')
#np.set_printoptions(threshold='nan')

def drawROC_fpr(**kwargs):
    """
    Input:{'model_name':[fpr, tpr, thresholds, val_info]}
    """
    fig,ax = plt.subplots( nrows=1, ncols=1 )
    legend_text=[]
    for key, value in kwargs.iteritems():
        auc_score = auc(value[0], value[1], reorder=True)
        print("AUC on test %s: %f" % (key, auc_score))
        ax.plot(value[0], value[1])
        legend_text.append("%s: %.3f, val: %.3f" % (key, auc_score, value[-1][1]))
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.legend(legend_text, loc='best')
    plt.grid(True)
    fig.savefig("ROC_fpr.png")
    plt.close(fig)
    #plt.show()
    return

def build_dict(**model_name):
    model_dict={}
    for ilabel, iname in model_name.iteritems():
        if (len(glob.glob("Experiments/"+iname+"/file_fpr")) != 1):
               raise StandardError('There should be ONE and ONLY ONE ROC for each model!!!')
        else:
            fpr=pickle.load(open(glob.glob("Experiments/"+iname+"/file_fpr")[0] ) ) 
            tpr=pickle.load(open(glob.glob("Experiments/"+iname+"/file_tpr")[0] ) )
            threshold=pickle.load(open(glob.glob("Experiments/"+iname+"/file_threshold")[0] ) )
            val_info=pickle.load(open(glob.glob("Experiments/"+iname+"/file_val")[0] ) ) 
            model_dict[ilabel]=[fpr, tpr, threshold, val_info]

    return model_dict

if __name__ == '__main__':

    # model_key = ['tree_tth_truth_tt_12th',
    #            'tree_tth_truth_tt_best',
    #            'tree_tth_best_tt_best',
    #            'dnn_tth_truth_tt_12th',
    #            'dnn_tth_truth_tt_best',
    #            'dnn_tth_best_tt_best',
    #            'bdt_w/o_btag']
    
    # model_value = ['one_tree',
    #              'tree_tt_best',
    #              'tree_tth_best_tt_best',
    #              'dnn',
    #              'dnn_tt_best',
    #              'dnn_tth_best_tt_best',
    #              'bdt_wo_btag']

    # model_key = ['w_lambda',
    #              'wo_lambda',
    #              'q_20',
    #              'dropout_all0.5',
    #              '80%_train',
    #              'input_withH',
    #              'bdt_wo_btag']
    
    # model_value = ['one_tree',
    #                'tree_no_lambda',
    #                'tree_with_lambda_lowQ',
    #                'tree_with_lambda_dropout',
    #                'tree_large_trainsize',
    #                'tree_input_withH',
    #                'bdt_wo_btag'
    #                ]

    model_key = ['tree_wo_btag',
                 'tree_w_btag',
                 'tree_w_btag_m_inLambda',
                 'tree_w_btag_ptEtaPhiE_inLambda',
                 'tree_w_btag_noLambda',
                 'dnn_w_btag',
                 'bdt_w_btag']
    
    model_value = ['tree_tth_best_tt_best',
                   'tree_tth_best_tt_best_withBTag',
                   'tree_tth_best_tt_best_withBTag_withM',
                   'tree_tth_best_tt_best_withBTag_lambda4d',
                   'tree_tth_best_tt_best_withBTag_noLambda',
                   'dnn_tth_best_tt_best_withBTag',
                   'bdt_w_btag'
                   ]


    model_name = {}
    for key, value in zip(model_key, model_value):
        model_name[key] = value

    model_dict = build_dict(**model_name)
    
    drawROC_fpr(**model_dict)
