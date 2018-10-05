import sklearn.metrics
from keras.callbacks import Callback

class roc_cb_earlyStop(Callback):

    def __init__(self, X_train, X_test, Y_train, Y_test, weight_train, weight_test, save_file=True, save_best_only=True, filepath="Weights-improvement-{epoch:02d}-{auc:.3f}.h5", weight_file="Weight-{epoch:02d}-{auc:.3f}.h5", early_stop=True, patience=10, verbose=0):
        super(roc_cb_earlyStop, self).__init__()
            
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.weight_train = weight_train
        self.weight_test = weight_test
        self.save_file = save_file
        self.save_best_only = save_best_only
        self.filepath = filepath
        self.weight_file = weight_file
        self.early_stop = early_stop
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_train_begin(self, logs={}):
        self.auc_train = []
        self.auc_test = []

    def on_epoch_end(self, epoch, logs={}):
        if(epoch==0):
            max_auc=0.
        else:
            max_auc=max(self.auc_test)

        logs = logs or {}
        ypred_train = self.model.predict(self.X_train, verbose=0).T[0]
        self.auc_train.append(sklearn.metrics.roc_auc_score(self.Y_train, ypred_train, sample_weight=self.weight_train))
        ypred_test = self.model.predict(self.X_test, verbose=0).T[0]
        self.auc_test.append(sklearn.metrics.roc_auc_score(self.Y_test, ypred_test, sample_weight=self.weight_test))

        if (self.save_file):
            filepath = self.filepath.format(epoch=epoch, auc=self.auc_test[-1])
            weight_file = self.weight_file.format(epoch=epoch, auc=self.auc_test[-1])
            if (self.save_best_only):
                if (self.auc_test[-1] > max_auc):
                    self.model.save(filepath, overwrite=True)
                    self.model.save_weights(weight_file)

                    print self.auc_test[-1]
                    print filepath
            else:
                self.model.save(filepath, overwrite=True)
                self.model.save_weights(weight_file)
                    
        if self.early_stop and (epoch>1):
            if(self.auc_test[-1] < max(self.auc_test)):
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                else:
                    self.wait += 1
            else:
                self.wait = 0

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))
