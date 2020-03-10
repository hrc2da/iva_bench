from models import Model, LossHistory, custom_metrics
from keras.models import Model as KModel
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, LSTM, Permute,Masking, Input, BatchNormalization, GlobalAveragePooling1D, Conv1D, Conv2D, Conv3D, MaxPooling1D, AveragePooling1D
from keras.layers import concatenate

'''
This implements multivariate lstm-fcn for time series classification
References:
https://arxiv.org/pdf/1709.05206.pdf (original lstm-fcn)

https://arxiv.org/abs/1801.04503 (multivariate version)
https://github.com/titu1994/MLSTM-FCN

'''

from copy import deepcopy

class LSTM_FCN(Model):
    def setup_model(self,specs):
        '''
            Adapted directly from generate_model here: https://github.com/titu1994/MLSTM-FCN/blob/master/acvitivity_model.py
        '''
        if 'pretrained_model' in specs:
            self.model = load_model(specs['pretrained_model'], custom_objects = custom_metrics)
            for layer in self.model.layers:
                layer_name = layer.name
                print("Overwriting layer {} name".format(layer_name))
                self.model.get_layer(layer_name).name = "old_"+layer_name
        
        #ip = Input(shape=(3, 20))
        ip = Input(shape=(20, 3))
        x = Permute((2,1))(ip)
        x = Masking()(ip)
        x = LSTM(8)(x)
        x = Dropout(0.8)(x)

        #y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        #y = squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        #y = squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(7, activation='softmax')(x)

        self.model = KModel(ip, out)
        self.model.compile(loss=specs['loss'],optimizer=specs['optimizer'],metrics=specs['metrics'])
        self.model.summary()
        

        # add load model code here to fine-tune

        

    def set_params(self,specs):
        #self.specs = deepcopy(specs) # Not used for now, but may do this later. (problem w/ deepcopy and objects in specs, really need to get objects out of specs)
        self.setup_model(specs)
    
    def fit(self,x,Y,fit_params=None):
        if fit_params is None:
            return self.model.fit(x,Y)
        else:
            return self.model.fit(x,Y,**fit_params)

    def save(self,path):
        return self.model.save(path)

    def evaluate(self,x,Y):
        return self.model.evaluate(x,Y)

    def predict(self,x):
        return self.model.predict(x)

    def predict_classes(self,x):
        return self.model.predict_classes(x)