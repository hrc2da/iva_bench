from keras import callbacks
from keras import backend as K
from sklearn.metrics import r2_score

class Model:
    def set_params(self,specs):
        raise(NotImplementedError)
    def setup_model(self):
        raise(NotImplementedError)
    def fit(self):
        raise(NotImplementedError)


# custom metrics
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ),axis=0) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true, axis=0) ),axis=0 ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def skl_r2score(y_true, y_pred):
    import pdb; pdb.set_trace()
    ytrue = K.eval(y_true)
    ypred = K.eval(ypred)
    return r2_score(ytrue, ypred)


custom_metrics = {
    'coeff_determination': coeff_determination,
    'skl_r2score': skl_r2score
}

# some callbacks
class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# add any models you write to this list and __all__
import models.keras_nn
__all__ = ['keras_nn']
