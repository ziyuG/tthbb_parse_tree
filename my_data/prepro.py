import numpy as np
#import prepare

def learn_val_split(m_array, m_ratio=0.2):
    m_learn = m_array[int(m_ratio*len(m_array)): ]
    m_val = m_array[: int(m_ratio*len(m_array))]
    return m_learn, m_val


def scale_norm(X_train, X_test, X_validation=None):
    """
    Then do scale and normalization.
    """
    
    X_learn = np.copy(X_train)
    X_app = np.copy(X_test)

    # ----------- Preprocessing -------------
    if(len(X_train.shape)==3):
        #Scale
        l_mean = np.mean(X_learn, axis = (0,1) ) # center
        #Normalisation of inputs.
        l_std = np.std(X_learn, axis = (0,1) ) # normalize
    elif(len(X_train.shape)==2):
        #Scale
        l_mean = np.mean(X_learn, axis = 0) # center
        #Normalisation of inputs.
        l_std = np.std(X_learn, axis = 0 ) # normalize

    l_std[l_std==0.] = 1.
    print l_std
    X_learn -= l_mean 
    X_learn /= l_std
    
    X_app -= l_mean
    X_app /= l_std

    X_val = np.copy(X_validation)
    if(X_validation is not None):
        X_val -= l_mean
        X_val /= l_std
        
    return X_learn, X_app, X_val, l_mean, l_std
