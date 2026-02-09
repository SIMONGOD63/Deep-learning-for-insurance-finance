#%% 0 packages
import pandas as pd ; import numpy as np
import tensorflow as tf ;from tensorflow import keras ; from keras.layers import Dense, Dropout ; from keras_tuner import Hyperband, Objective
from keras.optimizers import Adam ;from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from statistics import variance
from copy import deepcopy
from collections import Counter
import datetime


file_path = "C:/Users/User/Documents/Data Science/LDATS2310 DS for insu and finance/"

#%% Model found
def l_mod1(inpt = 27): # M4 with a LR
    
    model = keras.Sequential([
            Dense(144, 'tanh',kernel_initializer='he_normal', input_shape = (inpt,)), #☺
            Dense(112, 'tanh',kernel_initializer='he_normal'),

            Dense(name = 'response', units = 1,activation = "sigmoid")
            ])
        
    model.compile(loss = deviance,
                   optimizer = keras.optimizers.Adam(),
                   metrics = ['accuracy',keras.metrics.Recall()])
    
    return model



def l_mod2(inpt = 27): # M4 with a LR
    
    model = keras.Sequential([
            Dense(208, 'tanh',kernel_initializer='he_normal', input_shape = (inpt,)), #☺
            Dense(112, 'tanh',kernel_initializer='he_normal'),
            Dropout(0.5),
            
            Dense(128, 'tanh',kernel_initializer='he_normal'),
            Dense(240, 'tanh',kernel_initializer='he_normal'),
            Dense(192, 'tanh',kernel_initializer='he_normal'),
            Dropout(0.3),
            Dense(16, 'tanh',kernel_initializer='he_normal'),
            
            Dense(name = 'response', units = 1,activation = "sigmoid")
            ])
        
    model.compile(loss = deviance,
                   optimizer = keras.optimizers.Adam(),
                   metrics = ['accuracy',keras.metrics.Recall()])
    
    return model

def l_mod3(inpt = 27): # M3, no LR
    
    model = keras.Sequential([
            Dense(128, 'tanh',kernel_initializer='he_normal', input_shape = (inpt,)), #☺
            Dense(144, 'tanh',kernel_initializer='he_normal'),
            Dense(96, 'tanh',kernel_initializer='he_normal'),
            Dropout(0.2),
            
            Dense(224, 'tanh',kernel_initializer='he_normal'),
            Dropout(0.4),
            Dense(name = 'response', units = 1,activation = "sigmoid")
            ])
        
    model.compile(loss = deviance,
                   optimizer = keras.optimizers.Adam(clipvalue = 1),
                   metrics = ['accuracy'])
    
    return model

def l_mod4(inpt = 27): # M3, no LR
    
    model = keras.Sequential([
            Dense(224, 'tanh',kernel_initializer='he_normal', input_shape = (inpt,)), #☺
            Dense(144, 'tanh',kernel_initializer='he_normal'),
            Dense(144, 'tanh',kernel_initializer='he_normal'),
            Dense(240, 'tanh',kernel_initializer='he_normal'),
            Dense(name = 'response', units = 1,activation = "sigmoid")
            ])
        
    model.compile(loss = deviance,
                   optimizer = keras.optimizers.Adam(clipvalue = 1),
                   metrics = ['accuracy'])
    
    return model
#c_weight2 = {0 : 0.6834631613112626, 1 :1.862671384343211}

#%% 1 Load data and  Preprocessing

car_insu = pd.read_excel(file_path + "01Data/Car Insurance Claim.xlsx",index_col="ID")
data = deepcopy(car_insu)
print(data.shape)

# Try dropping Claim Amount cuz it gives too much information
data.drop(columns = ['Claims Amount'], inplace = True)


#-- Transform categorical variable into binary one
data['City Population'].value_counts()
data['License Revoked'].value_counts()
data['Car Use'].value_counts()
# get 01
data["Crash"] = (data['Claims Flag (Crash)'] == "Yes").astype("int64")
data['Urb'] = (data['City Population'] == "Urban").astype("int64")
data["Lics R"] = (data['License Revoked'] == "Yes").astype("int64")
data["Red"] = (data['Red Car?'] == "Yes").astype("int64")

data["Car Use"] = (data['Car Use'] == "Private").astype("int64")
data["Gender"] = (data['Gender'] == "M").astype("int64")
data["Marital Status"] = (data['Marital Status'] == "Yes").astype("int64")
data["Sing. Par"] = (data['Single Parent?'] == "Yes").astype("int64")

# multilevel categorical variable
to_categorical = ["Education","Occupation","Car Type"]
data[to_categorical] = data[to_categorical].astype('category')

#extract month of birth
data['Month'] = data['DOB'].dt.month

to_supp = ['City Population',"Claims Flag (Crash)","License Revoked","Red Car?","Single Parent?"]
data = data.drop(to_supp, axis =1)
del(to_supp)


#%% 2 Check for correlated variable

data.dtypes
num = data.select_dtypes(include = ['number'])
corr_num = num.corr()

upper_triangle_mask = np.triu(np.ones(corr_num.shape), k=1).astype(bool)
tri_CM = corr_num.where(upper_triangle_mask)

#------ 

high_corr_mask = abs(tri_CM) >= 0.6

# the loop remove the higly correlated variable from col_names, variable's name list
col_names =corr_num.columns.tolist()
for i in range(len(high_corr_mask)):
    for col_name,value in high_corr_mask.iloc[i].items():
        if value:
            if col_name in col_names:
                print(col_name," has been removed")
                col_names.remove(col_name)


# get rid of the correlated variables
data = data[[*col_names,*to_categorical,"Month"]]
del(i,col_name,value,high_corr_mask,tri_CM,num,upper_triangle_mask,col_names)
#%% Keep education and remove Occupation 
data.drop(['Occupation'], axis = 1, inplace = True)


e_dum = (pd.get_dummies(data["Education"], drop_first= True)).astype(int)
c_dum = (pd.get_dummies(data['Car Type'], drop_first = True)).astype(int)
month_dum = (pd.get_dummies(data['Month'], drop_first = True)).astype(int)

#  add those columns to the dataset
data = pd.concat([data,e_dum,c_dum,month_dum], axis = 1)

data.drop(["Education","Month","Car Type"], axis = 1, inplace = True) 
del(month_dum, to_categorical)
#%% 4 Analyze the variability and scale variable

variance = data.var()
std = data.std()
a = variance > 2
to_scale = list(variance[a].index)
scaler = StandardScaler().set_output(transform='pandas')
scaled_var = scaler.fit_transform(data[to_scale])
data[to_scale] = scaled_var

# check
data.var()

del(scaled_var,scaler,std, variance,a, to_scale)

#%% 5 CREATING BALANCED training sets 1 ONE
X = data.drop("Crash",axis = 1)
Y = data['Crash']
#convert them into array that's what I've never DONE BEFORE THE DEADLINE we are the 2/12
X = np.asarray(X)
Y = np.asarray(Y,dtype = np.float64)


# check the proportion of Crash
print(data['Crash'].value_counts() / len(data))


# creating train and tests sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=4243)

## create balanced train sets
smote = SMOTE(random_state=4243)
x_train,y_train = smote.fit_resample(x_train, y_train)


#need to shuffle them
x_train,y_train = shuffle(x_train,y_train,random_state=4243)

#test to see the proportions
Counter(y_train)
Counter(y_test)
del(smote)

#%% 8 Needed for NN
# Get the input size for the 1st layer


def deviance(y, p):
    eps = 1e-16
    y = tf.cast(y, dtype=tf.float32)
    #p = tf.convert_to_tensor(p, dtype=tf.float32)
    p = tf.clip_by_value(p, eps, 1 - eps)  # Clip predictions to avoid log of zero
    return -2*1 * tf.reduce_sum(y * tf.math.log((p + eps) / (1 - p + eps)) + tf.math.log(1 - p + eps))

    
#--- Re definition BCR function
def BCR(y_true, y_pred): # ????
    if y_true is None or y_true.shape[0] == 0:
        print(y_true)
        return 0.0
    # reshaping
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred > 0.5, dtype='float32')  # Convert probabilities to binary labels
    #print(y_pred)
    correct_preds = K.cast(K.equal(y_true, y_pred), 'float32')
    
    sum_per_class = K.sum(y_true, axis=0) + K.sum(1 - y_true, axis=0)  # Sum of true labels per class
    
    # Compute accuracy per class
    acc_per_class = K.sum(correct_preds * y_true, axis=0) / K.maximum(K.sum(y_true, axis=0), 1)  # Accuracy for class 1
    acc_per_class += K.sum(correct_preds * (1 - y_true), axis=0) / K.maximum(K.sum(1 - y_true, axis=0), 1)  # Accuracy for class 0
    acc_per_class /= 2  # Average the accuracies
    
    # Compute BCR
    BCR = K.mean(acc_per_class)
    return BCR


def BCR_GPT(y_true, y_pred):
    # Ensure y_true and y_pred are tensor and have correct shape
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred > 0.5, dtype='float32')  # Convert probabilities to binary labels
    
    correct_preds = K.cast(K.equal(y_true, y_pred), 'float32')
    
    # Sum of true labels per class
    sum_per_class = K.sum(y_true) + K.sum(1 - y_true)
    
    # Compute accuracy per class
    acc_class_1 = K.sum(correct_preds * y_true) / K.maximum(K.sum(y_true), 1)  # Accuracy for class 1
    acc_class_0 = K.sum(correct_preds * (1 - y_true)) / K.maximum(K.sum(1 - y_true), 1)  # Accuracy for class 0
    acc_per_class = (acc_class_1 + acc_class_0) / 2  # Average the accuracies
    
    return acc_per_class



# =============================================================================
# #to launch tensorboard # ???
# import subprocess
# log_dir = "C:/Users/User/Documents/Data Science/LINFO2262 ML/A5_Competition/DNN/logs/fit2/"
# command = ["tensorboard","--logdir",log_dir]
# subprocess.run(command, check = True)
# 
# =============================================================================

def build_model1(hp):
   
    model = keras.Sequential()
    model.add(Dense(hp.Int("n_init", min_value = 16, max_value = 256, step = 16), 
                    activation = 'tanh', kernel_initializer='he_normal'))
    
    
    # hidden layers    
    for lay in range(hp.Int("n_layer", min_value = 1, max_value = 5,step = 1)):
        activ = hp.Choice(f"activation_{lay}",['relu','tanh',"softplus"])
        units = hp.Int(f"n_{lay}", min_value=16, max_value=256, step=16)
        if units:
            model.add(Dense(units = units, activation = 'tanh', kernel_initializer='he_normal'))
        
        if hp.Boolean(f'dropout_{lay}') and units > 0:
            r = hp.Float(f'rate_d{lay}',min_value = 0.1, max_value = 0.5, step = 0.1)
            model.add(Dropout(rate = r))
    
    model.add(Dense(1,activation = 'sigmoid', name ="response"))

    model.compile(optimizer = Adam(),
                  loss = deviance,
                  metrics = ["accuracy",keras.metrics.Recall()])
    return model

def build_model2(hp):
   
    model = keras.Sequential()
    activ2 = hp.Choice("activation000",['relu','tanh',"softplus","selu"])
    model.add(Dense(hp.Int("n_init", min_value = 16, max_value = 256, step = 16), 
                    activation = activ2, kernel_initializer='he_normal'))
    
    
    # hidden layers    
    for lay in range(hp.Int("n_layer", min_value = 1, max_value = 5,step = 1)):
        activ = hp.Choice(f"activation_{lay}",['relu','tanh',"softplus","selu"])
        units = hp.Int(f"n_{lay}", min_value=16, max_value=256, step=16)
        if units:
            model.add(Dense(units = units, activation = activ, kernel_initializer='he_normal'))
        
        if hp.Boolean(f'dropout_{lay}') and units > 0:
            r = hp.Float(f'rate_d{lay}',min_value = 0.1, max_value = 0.5, step = 0.1)
            model.add(Dropout(rate = r))
    
    model.add(Dense(1,activation = 'sigmoid', name ="response"))
    
    optim = hp.Choice("optim", ['adam','rmsprop'])
    model.compile(optimizer = optim,
                  loss = deviance,
                  metrics = ["accuracy",keras.metrics.Recall()])
    return model

# get the class weights
y_enco = data['Crash']


# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_enco),
    y=y_enco
)
# Convert class weights to a dictionary
class_w2= dict(enumerate(class_weights))

#---- alternative way of computing class weights

classes, count = np.unique(Y, return_counts=True)
class_freq = count/len(Y)
#compute inverse of class frequencies
class_w1 = dict(enumerate(1 / class_freq))

del(class_freq, class_weights,y_enco,classes,count)

#%% 11 MOD1

tuner1 = Hyperband(
    build_model2,
    objective = Objective("val_loss", direction = "min"),
    max_epochs = 50,
    seed = 42434445,
    hyperband_iterations = 3,
    directory = file_path + "New_logs/last",
    project_name ="NNl1",
    max_consecutive_failed_trials= 4
    )

log_dir = file_path + "New_logs/last/NNl1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

tuner1.search(x_train,y_train,
              validation_data = (x_test,y_test),
              epochs=50, 
             batch_size= 256, #arbitraly chosen
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5,restore_best_weights = True, mode = "min"),
                        tensorboard_callback])
#Get the hyperparameters and build a model
best_hps_val1 = tuner1.get_best_hyperparameters(num_trials=1)[0].values
best_hps1 = tuner1.get_best_hyperparameters(num_trials=1)[0]
model1 = tuner1.hypermodel.build(best_hps1)

#%%  12 NN with DS 2

tuner2 = Hyperband(
    build_model1,
    objective = Objective("val_loss", direction = "min"),
    max_epochs = 50,
    seed = 42434445,
    hyperband_iterations = 3,
    directory = file_path + "New_logs/last",
    project_name ="NNl2",
    max_consecutive_failed_trials= 4
    )

log_dir = file_path + "New_logs/last/NNl2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

tuner2.search(x_train,y_train,
              validation_data = (x_test,y_test),
              epochs = 50, 
             batch_size= 512, #arbitraly chosen
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5,restore_best_weights = True, mode = "min"),
                        tensorboard_callback])

#Get the hyperparameters and build a a model
best_hps_val2 = tuner2.get_best_hyperparameters(num_trials=1)[0].values
best_hps2 = tuner2.get_best_hyperparameters(num_trials=1)[0]
model2 = tuner2.hypermodel.build(best_hps2)




#%% 13 NN with DS3

tuner3 = Hyperband(
    build_model1,
    objective = Objective("val_loss", direction = "min"),
    max_epochs = 50,
    seed = 42434445,
    hyperband_iterations = 3,
    directory = file_path + "New_logs/last",
    project_name ="NNl3",
    max_consecutive_failed_trials= 4
    )

log_dir = file_path + "New_logs/post_smotenc/NN2_test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

tuner3.search(x_train,y_train,
              validation_data = (x_test,y_test),
              epochs = 50, 
             batch_size= 256, #arbitraly chosen
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5,restore_best_weights = True, mode = "min"),
                        tensorboard_callback])

#Get the hyperparameters and build a model
best_hps_val3 = tuner3.get_best_hyperparameters(num_trials=1)[0].values
best_hps3 = tuner3.get_best_hyperparameters(num_trials=1)[0]
model3 = tuner3.hypermodel.build(best_hps3)


#-- Model 4
best_hps_val4 = tuner3.get_best_hyperparameters(num_trials=2)[1].values
best_hps4 = tuner3.get_best_hyperparameters(num_trials=2)[1]
model4 = tuner3.hypermodel.build(best_hps4)

#%% inspect model and test my models
for layer in model1.layers:
    print('-'*40)
    print(layer.get_config(),"\n")
    
model1.summary()
for layer in model2.layers:
    print('-'*40)
    print(layer.get_config(),"\n")

for layer in model3.layers:
    print('-'*40)
    print(layer.get_config(),"\n")
    
for layer in model4.layers:
    print('-'*40)
    print(layer.get_config(),"\n")

print(model1.optimizer)
print(model2.optimizer)
print(model3.optimizer)
print(model4.optimizer)
# Create lists for models, hyperparameters, and storing results
models = [tuner1.hypermodel.build(best_hps1), 
          tuner2.hypermodel.build(best_hps2), 
          tuner3.hypermodel.build(best_hps3), 
          tuner3.hypermodel.build(best_hps4)]

stocks = [[] for _ in range(4)]  # Initialize lists to store results for each model

# Loop through the process multiple times
for _ in range(15):
    for i, model in enumerate(models):
        # Fit the model
        model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  epochs=100,
                  class_weight=class_w2,
                  verbose=0,
                  batch_size=512,
                  callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=15,
                                                              restore_best_weights=True, mode="min")])
        # Make predictions
        pred = model.predict(x_test)
        dev = deviance(y_test, pred)
        
        # Convert predictions to binary
        y_pred = (pred > 0.5).astype(int)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        
        # Calculate BCR and store results
        stocks[i].append((BCR(y_test, pred).numpy(), acc, dev.numpy()))

print('ITS DONE MATES', "\n", "*" * 40)
print(i)
print(model)

for i in range(len(stocks)):
    p_bcr = [bcr[0] for bcr in stocks[i]]
    p_accu = [bcr[1] for bcr in stocks[i]]
    p_loss = [it[2] for it in stocks[i]]
    print(f'M{i + 1} : loss : {np.mean(p_loss)}, acc : {np.mean(p_accu)} , bcr : {np.mean(p_bcr)} \n')


#%% saving the best model
last_mod = tuner2.hypermodel.build(best_hps2) #l_mod1() 
last_mod.fit(x_train,y_train,
          validation_data = (x_test,y_test),
          epochs = 100,
          class_weight = class_w2,
          batch_size = 512,
          verbose = 0,
          callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5,
                                                      restore_best_weights = True, mode = "min",
                                                      verbose = 0)])
pred = (last_mod.predict(x_test) > 0.5).astype(int)
accuracy_score(y_test, pred)
BCR(y_test, pred)
last_mod.summary()
#last_mod.save_weights('C:/Users/User/Documents/Data Science/LDATS2310 DS for insu and finance/Mod_weights/mod_lastxxx.weights.h5')

#last_mod.laod_weights('C:/Users/User/Documents/Data Science/LDATS2310 DS for insu and finance/mod.weights.h5')

#%% --- Model found with hyperband tuner
def l_mod1(inpt = 27): 
    
    model = keras.Sequential([
            Dense(144, 'tanh',kernel_initializer='he_normal', input_shape = (inpt,)), #☺
            Dense(112, 'tanh',kernel_initializer='he_normal'),

            Dense(name = 'response', units = 1,activation = "sigmoid")
            ])
        
    model.compile(loss = deviance,
                   optimizer = keras.optimizers.Adam(),
                   metrics = ['accuracy',keras.metrics.Recall()])
    
    return model



def l_mod2(inpt = 27): 
    
    model = keras.Sequential([
            Dense(208, 'tanh',kernel_initializer='he_normal', input_shape = (inpt,)), #☺
            Dense(112, 'tanh',kernel_initializer='he_normal'),
            Dropout(0.5),
            
            Dense(128, 'tanh',kernel_initializer='he_normal'),
            Dense(240, 'tanh',kernel_initializer='he_normal'),
            Dense(192, 'tanh',kernel_initializer='he_normal'),
            Dropout(0.3),
            Dense(16, 'tanh',kernel_initializer='he_normal'),
            
            Dense(name = 'response', units = 1,activation = "sigmoid")
            ])
        
    model.compile(loss = deviance,
                   optimizer = keras.optimizers.Adam(),
                   metrics = ['accuracy',keras.metrics.Recall()])
    
    return model

def l_mod3(inpt = 27): # M3, no LR
    
    model = keras.Sequential([
            Dense(128, 'tanh',kernel_initializer='he_normal', input_shape = (inpt,)), #☺
            Dense(144, 'tanh',kernel_initializer='he_normal'),
            Dense(96, 'tanh',kernel_initializer='he_normal'),
            Dropout(0.2),
            
            Dense(224, 'tanh',kernel_initializer='he_normal'),
            Dropout(0.4),
            Dense(name = 'response', units = 1,activation = "sigmoid")
            ])
        
    model.compile(loss = deviance,
                   optimizer = keras.optimizers.Adam(clipvalue = 1),
                   metrics = ['accuracy'])
    
    return model

def l_mod4(inpt = 27): # M3, no LR
    
    model = keras.Sequential([
            Dense(224, 'tanh',kernel_initializer='he_normal', input_shape = (inpt,)), #☺
            Dense(144, 'tanh',kernel_initializer='he_normal'),
            Dense(144, 'tanh',kernel_initializer='he_normal'),
            Dense(240, 'tanh',kernel_initializer='he_normal'),
            Dense(name = 'response', units = 1,activation = "sigmoid")
            ])
        
    model.compile(loss = deviance,
                   optimizer = keras.optimizers.Adam(clipvalue = 1),
                   metrics = ['accuracy'])
    
    return model
