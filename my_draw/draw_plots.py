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

#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True

#np.seterr(divide='ignore', invalid='ignore')
#np.set_printoptions(threshold='nan')

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print cm
    #plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black") #j is x axis position, i is y axis position
        #print("text position: %i %i in data coordinates" % (j, i))
        #print cm[i,j]
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#The abs weight value is used, because use the confusion matrix as a meature of model performance.
def draw_confusion_matrix(Y_test, p_test, sample_weight_test, class_tag): #class_tag = ['tth', 'ttb', 'ttc']
    labels = range(len(class_tag))
    # Updated snippet !!! Only read in prediction and truth label array (not label matrix).
    Y_test_classes = Y_test.argmax(axis=-1)
    p_test_classes = p_test.argmax(axis=-1)
    #Draw confution matrix
    conf_mat = confusion_matrix(Y_test_classes, p_test_classes, labels=labels, sample_weight=sample_weight_test)
    
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=class_tag, normalize=False)
    plt.savefig('confusionMatrix_noNorm.png')
    
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=class_tag, normalize=True)
    plt.savefig('confusionMatrix_Norm.png')


def flavorTransfor_4c(flavorType):

    if(flavorType=='tth'):
        flavorN=0
    elif(flavorType=='ttb'):
        flavorN=1
    elif(flavorType=='ttc'):
        flavorN=2
    elif(flavorType=='ttl'):
        flavorN=3
    return flavorN

def flavorTransfor(flavorType):

    if(flavorType=='tth'):
        flavorN=0
    elif(flavorType=='ttb_ttb'):
        flavorN=1
    elif(flavorType=='ttb_ttB'):
        flavorN=2
    elif(flavorType=='ttb_ttbb'):
        flavorN=3
    elif(flavorType=='ttc'):
        flavorN=4
    elif(flavorType=='ttl'):
        flavorN=5
    else:
        print("ERROR: the input flavor name is not correct!")
        return 
    return flavorN


def drawTestOutput(p_test, Y_test, weight_test, xmin, xmax, num_bins, node, class_tag): #class_tag = ['tth', 'ttb', 'ttc']
    """
    For each node, draw test output
    """
    plt.figure()


    d_bins = (xmax-xmin) / float(num_bins)

    nNode=flavorTransfor(node)
    x_l=0.
    x_h=1.
    # the histogram of the data
    #test_sig
    evt_tot, cat = Y_test.shape
    
    for i_c in range(cat):
        ns, bins, patches = plt.hist(p_test[Y_test[:, i_c]==1][:, nNode], num_bins, range=(x_l, x_h), normed=1, histtype='step', weights=weight_test[Y_test[:, i_c]==1] )

        plt.grid(True)
        plt.legend(class_tag, loc='best')
        plt.title(node+" node")
        plt.savefig("RNNOutput4Class_"+node+".png")

    return


# def drawTestOutputCMS(p_test, Y_test, xmin, xmax, num_bins, node, class_tag):
#     """
#     Draw models output distribution in each region, defined a la CMS
#     """
#     plt.figure()
#     d_bins = (xmax-xmin) / float(num_bins)
#     p_test_classes = p_test.argmax(axis=-1)
    
#     nNode=flavorTransfor(node)
#     x_l=0.
#     x_h=1.

#     evt_tot, cat = Y_test.shape
    
#     for i_c in range(cat):
#         ns, bins, patches = plt.hist(p_test[(p_test_classes==nNode) & (Y_test[:, i_c]==1)][:, nNode], num_bins, range=(x_l, x_h), normed=1, histtype='step', weights=weight_test[(p_test_classes==nNode) & (Y_test[:, i_c]==1)] )

#         plt.grid(True)
#         plt.legend(class_tag, loc='best')
#         plt.title(node+" node")
#         plt.savefig("RNNOutput4Class_in_"+node+".png")

#     return


def calculatePie(nNode, p_test_classes, Y_test, weight_test):
    #nNode=flavorTransfor(node)
    frac=[]
    cat = Y_test.shape[1]
    
    for i_c in range(cat):
        frac.append( weight_test[(p_test_classes==nNode) & (Y_test[:, i_c]==1)].sum() / weight_test[(p_test_classes==nNode)].sum() )

    return frac


def calculatePie_weight(nNode, p_test_classes, Y_test, weight_test):
    #nNode=flavorTransfor(node)
    frac=[]
    evt_tot, cat = Y_test.shape
    
    for i_c in range(cat):
        frac.append( weight_test[(p_test_classes==nNode) & (Y_test[:, i_c]==1)].sum() ) #/ weight_test[(p_test_classes==nNode)].sum() )

    tot_weight = weight_test[(p_test_classes==nNode)].sum()
    frac.append(tot_weight)
    
    return frac


def drawPieCMS(p_test_classes, Y_test, weight_test, node, class_tag):
    font = {'family' : 'normal',
        'weight' : 'bold',
            'size'   : 10}

    #plt.figure()
    plt.figure(figsize=(4, 3))
    plt.axis("equal")
    plt.rc('font', **font)

    n_class = len(class_tag)
    if(n_class==6):
        nNode = flavorTransfor(node)
    elif(n_class==4):
        nNode = flavorTransfor_4c(node)
        
    #p_test_classes = p_test.argmax(axis=-1)

    frac = np.array(calculatePie(nNode, p_test_classes, Y_test, weight_test))
    frac = np.array(frac)
    data = frac[(frac > 1e-5)]

    arr_weight = np.array(calculatePie_weight(nNode, p_test_classes, Y_test, weight_test))
    labels = []
    for i_arr, i_c in zip(arr_weight, class_tag):
        i_lab = i_c + " \n%.1f" % i_arr
        labels.append(i_lab)
    labels = np.array(labels)
    labels = labels[(frac > 1e-5)]
        
    #labels= "tth \n%.1f" % arr_weight[0], "ttb %.1f" % arr_weight[1], "ttc %.1f" % arr_weight[2], "ttl %.1f" % arr_weight[3]
    colors_list = np.array(['b','orange','g','r', 'magenta', 'cyan', 'olive'])
    colors = colors_list[: 6]
    if(len(class_tag)==4):
        del_index = [2, 3]
        colors = np.delete(colors, del_index)

    colors = colors[(frac > 1e-5)]
 
    patches, texts, autotexts = plt.pie(data, colors=colors, labels=labels, autopct='%1.1f%%', shadow=False)
    
    if(nNode==0):
        s_sqrtb = weight_test[(p_test_classes==nNode) & (Y_test[:, 0]==1)].sum() / (weight_test[(p_test_classes==nNode) & (Y_test[:, 0]==0)].sum())**(1/2.0)
        s_b = weight_test[(p_test_classes==nNode) & (Y_test[:, 0]==1)].sum() / weight_test[(p_test_classes==nNode) & (Y_test[:, 0]==0)].sum()*100
        plt.title(node+" node, %.1f, " % arr_weight[-1] + r"$\frac{s}{\sqrt{b}}=$" + str(round(s_sqrtb, 2)) +", "+ r"$\frac{s}{b}=$" + str(round(s_b, 2))+'%')
    else:
        plt.title(node+" node %.1f" % arr_weight[-1] )
        
    plt.savefig("RNNOutput4Class_PIE_"+node+".png")
    return


# def drawTestOutputNodewise(p_test, Y_test, xmin, xmax, num_bins, flavorType):

#     plt.figure()


#     d_bins = (xmax-xmin) / float(num_bins)

#     x_l=0.
#     x_h=1.
#     flavorN=0
#     # the histogram of the data
#     if(flavorType=='tth'):
#         flavorN=0
#     elif(flavorType=='ttb'):
#         flavorN=1
#     elif(flavorType=='ttc'):
#         flavorN=2
#     elif(flavorType=='ttl'):
#         flavorN=3

#     #test_sig
#     #tth node
#     ns, bins, patches = plt.hist(p_test[Y_test[:, flavorN]==1][:, 0], num_bins, range=(x_l, x_h), # normed=1, 
#                                  histtype='step', weights=weight_test[Y_test[:, flavorN]==1] )
#     #ttb node
#     ns, bins, patches = plt.hist(p_test[Y_test[:, flavorN]==1][:, 1], num_bins, range=(x_l, x_h), # normed=1,
#                                  histtype='step', weights=weight_test[Y_test[:, flavorN]==1] )
#     #ttc node
#     ns, bins, patches = plt.hist(p_test[Y_test[:, flavorN]==1][:, 2], num_bins, range=(x_l, x_h), # normed=1,
#                                  histtype='step', weights=weight_test[Y_test[:, flavorN]==1] )
#     #ttl node
#     ns, bins, patches = plt.hist(p_test[Y_test[:, flavorN]==1][:, 3], num_bins, range=(x_l, x_h), # normed=1,
#                                  histtype='step', weights=weight_test[Y_test[:, flavorN]==1] )

#     plt.grid(True)
#     plt.legend(['tth_node', 'ttb_node', 'ttc_node', 'ttl_node'], loc='best')
#     plt.title(flavorType)
#     plt.savefig(flavorType+'.png')
#     return

#For each node, find the maximum across all the flavor samples
# def find_max_min(flavorN, p_test, Y_test):
    
#     m_max=max(p_test[Y_test[:, 0]==1][:, flavorN].max(), p_test[Y_test[:, 1]==1][:, flavorN].max(), p_test[Y_test[:, 2]==1][:, flavorN].max(), p_test[Y_test[:, 3]==1][:, flavorN].max())
#     m_min=min(p_test[Y_test[:, 0]==1][:, flavorN].min(), p_test[Y_test[:, 1]==1][:, flavorN].min(), p_test[Y_test[:, 2]==1][:, flavorN].min(), p_test[Y_test[:, 3]==1][:, flavorN].min())
    
#     return m_max, m_min


# def drawTestOutputNodewise2D(p_test, Y_test, weight_test, num_bins, flavorType1, flavorType2):

#     Nr=2
#     Nc=2
#     cmap = "cool"

#     fig, axs = plt.subplots(Nr, Nc)
#     fig.suptitle('2D plots in '+flavorType1+'_'+flavorType2+' node')
    
#     flavorN1=flavorTransfor(flavorType1)
#     flavorN2=flavorTransfor(flavorType2)
#     #Fix the x and y axis range
#     max1, min1 = find_max_min(flavorN1, p_test, Y_test)
#     max2, min2 = find_max_min(flavorN2, p_test, Y_test)

#     flavor_sample=np.array([['tth', 'ttb'],['ttc', 'ttl']])
#     images = []
#     for i in range(Nr):
#         for j in range(Nc):
            
#             flavorDN=flavorTransfor(flavor_sample[i][j])

#             #All the flavor samples are scaled to 1 to focus on the shape difference
#             h, hx, hy, h_image = axs[i,j].hist2d(p_test[Y_test[:, flavorDN]==1][:, flavorN1], p_test[Y_test[:, flavorDN]==1][:, flavorN2], bins=(num_bins, num_bins), range=([min1, max1], [min2, max2]), normed=1, weights=weight_test[Y_test[:, flavorDN]==1], cmap=cmap )

#             images.append(h_image)
#             axs[i, j].label_outer()
#             axs[i, j].set_title(flavor_sample[i][j])
#             axs[i, j].grid(True)
            
#             # Find the min and max of all colors for use in setting the color scale.
#             vmin = min(image.get_array().min() for image in images)
#             vmax = max(image.get_array().max() for image in images)
#             norm = colors.Normalize(vmin=vmin, vmax=vmax)

#     for im in images:
#         im.set_norm(norm)
                
#     fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

#     def update(changed_image):
#         for im in images:
#             if (changed_image.get_cmap() != im.get_cmap()
#                 or changed_image.get_clim() != im.get_clim()):
#                 im.set_cmap(changed_image.get_cmap())
#                 im.set_clim(changed_image.get_clim())

                
#     for im in images:
#         im.callbacksSM.connect('changed', update)
        
#     # plt.show()


#     # plt.colorbar()
#     #plt.grid(True)
#     # plt.legend(['tth', 'ttb'], loc='best')
#     # plt.title(flavorDraw)
#     # plt.xlabel(flavorType1)
#     # plt.ylabel(flavorType2)
#     plt.savefig(flavorType1+'_'+flavorType2+'.png')
#     plt.close(fig)
#     return


def drawROCCompare(Y_test, p_test, weight_test, ClassifBDTOutput_inclusive_withBTag_new_test):

    fpr1, tpr1, thresholds1 = roc_curve(Y_test[:, 0], p_test[:, 0], sample_weight=weight_test)
    fpr5, tpr5, thresholds5 = roc_curve(Y_test[:, 0], ClassifBDTOutput_inclusive_withBTag_new_test, sample_weight=weight_test)

    rnnScore = roc_auc_score(Y_test[:, 0], p_test[:, 0], sample_weight=weight_test)
    print("RNN AUC on test: %f" % rnnScore)
    bdtScore = roc_auc_score(Y_test[:, 0], ClassifBDTOutput_inclusive_withBTag_new_test, sample_weight=weight_test)
    print("BDT AUC on test: %f" % bdtScore)

    
    fig,ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(fpr1, tpr1)
    ax.plot(fpr5, tpr5)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.legend(["RNN: %.3f" % rnnScore, "classBDT: %.3f" % bdtScore], loc='best')
    plt.grid(True)
    fig.savefig("ROC_RNN_withH_allComb_withBTag_withClass_multiClass.png")
    plt.close(fig)
    #plt.show()
    return

def drawROC_pred(Y_test, p_test, weight_test, **kwargs):
    fig,ax = plt.subplots( nrows=1, ncols=1 )
    legend_text=[]
    fpr1, tpr1, thresholds1 = roc_curve(Y_test, p_test, sample_weight=weight_test)
    rnnScore = roc_auc_score(Y_test, p_test, sample_weight=weight_test)
    print("tree AUC on test: %f" % rnnScore)
    ax.plot(fpr1, tpr1)
    legend_text.append("tree: %.3f" % rnnScore)
    for key, value in kwargs.iteritems():
        fpr, tpr, thresholds = roc_curve(Y_test, value, sample_weight=weight_test)
        score = roc_auc_score(Y_test, value, sample_weight=weight_test)
        print("AUC on test %s: %f" % (key, score))
        ax.plot(fpr, tpr)
        legend_text.append("%s: %.3f" % (key, score))
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.legend(legend_text, loc='best')
    plt.grid(True)
    fig.savefig("ROC_pred.png")
    plt.close(fig)
    #plt.show()
    return

def drawROC_check(Y, Y_1, Y_2, p, p_1, p_2, weight, weight_1, weight_2, class_id):
    lw=2
    fpr, tpr, t = roc_curve(Y[:, class_id], p[:, class_id], sample_weight=weight)
    roc_auc = auc(fpr, tpr, reorder=True)

    fpr1, tpr1, t1 = roc_curve(Y_1[:, class_id], p_1[:, class_id], sample_weight=weight_1)
    roc_auc_1 = auc(fpr1, tpr1, reorder=True)

    fpr2, tpr2, t2 = roc_curve(Y_2[:, class_id], p_2[:, class_id], sample_weight=weight_2)
    roc_auc_2 = auc(fpr2, tpr2, reorder=True)

    fig,ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(fpr, tpr, lw=lw,
        label='ROC curve of tot (area = {0:0.3f})'.format(roc_auc))

    ax.plot(fpr1, tpr1, lw=lw,
        label='ROC curve of fold-1 (area = {0:0.3f})'.format(roc_auc_1))

    ax.plot(fpr2, tpr2, lw=lw,
        label='ROC curve of fold-2 (area = {0:0.3f})'.format(roc_auc_2))

    
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")

    plt.grid(True)
    fig.savefig("ROC_foldCheck_"+str(class_id)+".png")
    plt.close(fig)
    #plt.show()
    return

    
def drawROCCompare_multi(Y_test, p_test, weight_test, ClassifBDTOutput_inclusive_withBTag_new_test, class_tag):

    cat = Y_test.shape[1]
    lw=2
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(cat):
        print i
        fpr[i], tpr[i], t = roc_curve(Y_test[:, i], p_test[:, i], sample_weight=weight_test)
        roc_auc[i] = auc(fpr[i], tpr[i], reorder=True)


    # Compute micro-average ROC curve and ROC area
    weight_md = np.tile(np.transpose([weight_test]), cat)
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), p_test.ravel(), sample_weight=weight_md.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"], reorder=True)

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(cat)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(cat):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    # Finally average it and compute AUC
    mean_tpr /= cat
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"], reorder=True)

    # Plot all ROC curves
    #plt.figure()
    fig,ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.3f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    ax.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.3f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    bdtScore = roc_auc_score(Y_test[:, 0], ClassifBDTOutput_inclusive_withBTag_new_test, sample_weight=weight_test)
    print("BDT AUC on test: %f" % bdtScore)
    fpr5, tpr5, thresholds5 = roc_curve(Y_test[:, 0], ClassifBDTOutput_inclusive_withBTag_new_test, sample_weight=weight_test)
    ax.plot(fpr5, tpr5, 'k--', label='BDT ROC curve (area = {0:0.3f})'.format(bdtScore))

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'blue'])
    for i, color, iclass in zip(range(cat), colors, class_tag):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.3f})'.format(iclass, roc_auc[i]))


    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
  
    # for ic in range(cat):
    #     rnnScore = roc_auc_score(Y_test[:, ic], p_test[:, ic], sample_weight=weight_test)
    #     fpr, tpr, thresholds = roc_curve(Y_test[:, ic], p_test[:, ic], sample_weight=weight_test)
    #     ax.plot(fpr, tpr, label="rnn "+class_tag[ic]+": %.3f"% rnnScore)

    # print("RNN AUC on test: %f" % rnnScore)
    plt.grid(True)
    fig.savefig("ROC_multiClass.png")
    plt.close(fig)
    #plt.show()
    return


##################
#New ways to show the binary classification capacity: 
#Way one: give some confidence w.r.t tth node
def func_pred(t, label, pred, weight):
    """
    pred should be a 4-vector array, lable 1-vector array
    """
    pred_class=np.zeros(len(pred))

    for i in range(len(pred)):
        if(pred[i][0]>=t):
            if(pred[i].argmax(axis=-1)==0):
                iclass = 1
            elif(pred[i].argmax(axis=-1)!=0):
                iclass = 0
        elif(pred[i][0]<t):
            iclass = 0

        pred_class[i]=iclass

    tpr = weight[(pred_class==1) & (label==1)].sum() / weight[(label==1)].sum()
    fpr = weight[(pred_class==1) & (label==0)].sum() / weight[(label==0)].sum()
    return fpr, tpr

#Way two: give some confidence w.r.t bkg node
def func_pred2(t, label, pred, weight):
    """
    pred should be a 4-vector array, lable 1-vector array
    """
    pred_class=np.zeros(len(pred))

    for i in range(len(pred)):
        if(pred[i][1]>=t or pred[i][2]>=t or pred[i][3]>=t):
            if(pred[i].argmax(axis=-1)==0):
                iclass = 1
            elif(pred[i].argmax(axis=-1)!=0):
                iclass = 0
        elif( not(pred[i][1]>=t or pred[i][2]>=t or pred[i][3]>=t) ):
                iclass = 1

        pred_class[i]=iclass

    tpr = weight[(pred_class==1) & (label==1)].sum() / weight[(label==1)].sum()
    fpr = weight[(pred_class==1) & (label==0)].sum() / weight[(label==0)].sum()
    return fpr, tpr
    
# def func_pred_normal(t, true_class, pred_class, weight):
#     """
#     pred_class should be a 1-vector array, true_class 1-vector array
#     """
#     tpr = weight[(pred_class==1) & (true_class==1)].sum() / weight[(true_class==1)].sum()
#     fpr = weight[(pred_class==1) & (true_class==0)].sum() / weight[(true_class==0)].sum()
#     return fpr, tpr

def roc_cal(Y_test, p_test, weight_test, pred_func, out_range=[0., 1.], nbin=100):
    """
    This function is a self-defined roc, return fpr, tpr, and threshold arrays
    """
    t_arr = [x*(out_range[1]-out_range[0])/float(nbin) for x in range(nbin+1)]
    print t_arr
    
    fpr_arr=[]
    tpr_arr=[]

    #Could be in an more elegant way
    for i in range(len(t_arr)):
        fpr, tpr = pred_func(t_arr[i], Y_test, p_test, weight_test)
        fpr_arr.append(fpr)
        tpr_arr.append(tpr)

    print fpr_arr
    print tpr_arr
    return fpr_arr, tpr_arr, t_arr
        
# def drawROC_compare6and4(Y6, Y4, p6, p4, weight6, weight4, class_id):
#     lw=2
#     fpr, tpr, t = roc_curve(Y[:, class_id], p[:, class_id], sample_weight=weight)
#     roc_auc = auc(fpr, tpr, reorder=True)

#     fpr1, tpr1, t1 = roc_curve(Y_1[:, class_id], p_1[:, class_id], sample_weight=weight_1)
#     roc_auc_1 = auc(fpr1, tpr1, reorder=True)


def drawROC_trial(Y_test, p_test, weight_test, pred_func, name):#, ClassifBDTOutput_inclusive_withBTag_new_test):

    fpr5, tpr5, thresholds5 = roc_curve(Y_test[:, 0], p_test[:, 0], sample_weight=weight_test)
    fpr1, tpr1, thresholds1 = roc_cal(Y_test[:, 0], p_test, weight_test, pred_func)

    # rnnScore = roc_auc_score(Y_test[:, 0], p_test[:, 0], sample_weight=weight_test)
    # print("RNN AUC on test: %f" % rnnScore)
    # bdtScore = roc_auc_score(Y_test[:, 0], ClassifBDTOutput_inclusive_withBTag_new_test, sample_weight=weight_test)
    # print("BDT AUC on test: %f" % bdtScore)

    
    fig,ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(fpr5, tpr5, 'k--')
    ax.plot(fpr1, tpr1)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('bkg selection (false positive rate)')
    ax.set_ylabel('signal selection (true positive rate)')
    ax.set_title('ROC')
    ax.legend(['default_auc', 'new_auc'], loc='best')
    plt.grid(True)
    fig.savefig("ROC_RNN_trial_"+name+".eps")
    plt.close(fig)
    #plt.show()
    return
    
