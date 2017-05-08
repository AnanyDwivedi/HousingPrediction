import numpy,math
import keras,csv,os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_array
from keras.utils.visualize_util import plot
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History
#from keras import losses

from keras.layers import Activation
from keras.utils.np_utils import to_categorical
#from sklearn.metrics import accuracy_score
from keras import metrics
from keras import backend as K
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#def krmsle(actual, predicted):
#	m = Sequential()
#	m.compile(loss='mean_squared_logarithmic_error', optimizer=adam)
#	return m.mean_squared_logarithmic_error(actual, predicted)

def se(actual, predicted):
    """
    Computes the squared error.
    This function computes the squared error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The squared error between actual and predicted
    """
    return numpy.power(numpy.array(actual)-numpy.array(predicted), 2)

def mse(actual, predicted):
    """
    Computes the mean squared error.
    This function computes the mean squared error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The mean squared error between actual and predicted
    """
    return numpy.mean(se(actual, predicted))

def sle(actual, predicted):
    """
    Computes the squared log error.
    This function computes the squared log error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The squared log error between actual and predicted
    """
    return (numpy.power(numpy.log(numpy.array(actual)+1) - 
            numpy.log(numpy.array(predicted)+1), 2))


def msle(actual, predicted):
    """
    Computes the mean squared log error.
    This function computes the mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The mean squared log error between actual and predicted
    """
    return numpy.mean(sle(actual, predicted))

def rmsle(actual, predicted):
    """
    Computes the root mean squared log error.
    This function computes the root mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared log error between actual and predicted
    """
    return numpy.sqrt(msle(actual, predicted))

# learning rate schedule
def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 100.0
        lrate = initial_lrate * math.pow(drop,((1+epoch)/epochs_drop))
        return lrate


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
data = numpy.load('../data/PreProcessData.npy')
X = data[:,1:(data.shape[1]-1)]
Y = data[:,(data.shape[1]-1)]

val_data = numpy.load('../data/PreProcessDataTest.npy')
val = val_data[:,1:]

#PCA
#pca = PCA(n_components=100)
#X = pca.fit_transform(x)

# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
#X_train, X_val, y_train, y_val = data_split(X_train1, y_train1, test_size=0.1, random_state=seed)

#X_val = numpy.concatenate((X_test,X_val),axis=0)
#y_val = numpy.concatenate((y_test,y_val),axis=0)

#for i in range(X.shape[1]):
#        X_train[:,i]=numpy.divide(numpy.subtract(X_train[:,i],X_train[:,i].mean()),X_train[:,i].std())
#        X_val[:,i]=numpy.subtract(X_val[:,i],X_val[:,i].mean())
#        X_test[:,i]=numpy.divide(numpy.subtract(X_test[:,i],X_train[:,i].mean()),X_train[:,i].std())

print y_train.shape
print X_train.shape
# create model
model = Sequential()
prelu=keras.layers.advanced_activations.PReLU()
prelu1=keras.layers.advanced_activations.PReLU()
prelu2=keras.layers.advanced_activations.PReLU()
prelu3=keras.layers.advanced_activations.PReLU()
prelu4=keras.layers.advanced_activations.PReLU()
prelu5=keras.layers.advanced_activations.PReLU()
prelu6=keras.layers.advanced_activations.PReLU()
prelu7=keras.layers.advanced_activations.PReLU()
HiddenNeurons = 128


model.add(Dense(HiddenNeurons, input_dim=X_train.shape[1], init='uniform',name='h1', activation='relu'))
#model.add(prelu)
#model.add(Dropout(0.15))

model.add(Dense(HiddenNeurons, init='uniform', name='h2', activation='relu'))
#model.add(prelu1)
#model.add(Dropout(0.25))

model.add(Dense(HiddenNeurons, init='uniform',name='h3', activation='relu'))
#model.add(prelu2)
#model.add(Dropout(0.5))

model.add(Dense(HiddenNeurons, init='uniform',name='h4', activation='relu'))
#model.add(prelu3)
#model.add(Dropout(0.5))

model.add(Dense(HiddenNeurons, init='uniform',name='h5', activation='relu'))
#model.add(prelu4)
#model.add(Dropout(0.5))

model.add(Dense(HiddenNeurons, init='uniform',name='h6', activation='relu'))
#model.add(prelu5)
#model.add(Dropout(0.45))

model.add(Dense(HiddenNeurons, init='uniform',name='h7', activation='relu'))
#model.add(prelu6)
#model.add(Dropout(0.45))

model.add(Dense(HiddenNeurons, init='uniform',name='h8', activation='relu'))
#model.add(prelu7)
#model.add(Dropout(0.45))

model.add(Dense(1, init='uniform',name='out'))
#model.add(Activation('softmax'))

# Compile model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "FullyConnectedNetworkPrelu.png")
plot(model, to_file=model_path, show_shapes=True)
adam=keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_logarithmic_error', optimizer=adam,metrics=['msle'])

# learning schedule callback
history=History()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate,history]

#model Fitting
print "Training..."
model.fit(X_train, y_train,validation_data=(X_test,y_test),nb_epoch=5, batch_size=100, callbacks=callbacks_list, verbose=1)
#model.fit(X_train, y_train,validation_data=(X_test,y_test),nb_epoch=550, batch_size=X_train.shape[0],class_weight={0:1, 1:6756.0/271}, callbacks=callbacks_list, verbose=1)

#Model prediction
pred=model.predict(val,batch_size=25)
#err = model.metrics.mean_squared_logarithmic_error(y_test, pred)
#print err

numpy.save("Prediction.npy",pred)
#numpy.save("Xtest.npy",X_test)
#model.save('H8_student.h5')

