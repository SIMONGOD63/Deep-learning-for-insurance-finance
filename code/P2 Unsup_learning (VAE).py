#%% 0 packages
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.layers import Dense, Dropout, Input
from keras_tuner import Hyperband
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from statistics import variance
from copy import deepcopy
from collections import Counter
from scipy.stats import chi2_contingency

# VAE class
class VAE(keras.Model):
    # Class from LDATS2310
    def __init__(self, encoder, decoder,wgt, **kwargs): # I could implement a beta parameter to balanced the KL divergence  beta = 1.0
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.wgt     = wgt          #weights for the MSE
        #self.beta = beta            #parameter to balanced between the two losses
            
    def call(self, inputs):
       z_mean, z_log_var = self.encoder(inputs)
       z                 = self._sampling(z_mean, z_log_var)
       reconstruction    = self.decoder(z)
       loss              = self._VAE_loss(inputs,reconstruction, z_mean,z_log_var)
       self.add_loss(loss)
       return z, z_mean , z_log_var , reconstruction

    def _sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z
    #--- Method to load and save model
    def from_config(cls, config):
        encoder = config['encoder']
        decoder = config['decoder']
        wgt = config['wgt']
        return cls(encoder=encoder, decoder=decoder, wgt=wgt)
    
    def _VAE_loss(self,inputs,outputs,z_mean,z_log_var):
        #bin_loss            = tf.losses.BinaryCrossentropy()
        #reconstruction_loss = bin_loss(inputs,outputs)
        #reconstruction_loss = tf.losses.MSE(inputs,outputs)
        #reconstruction_loss =  0.5*tf.reduce_sum(tf.square((inputs-outputs)/self.wgt)) 
        reconstruction_loss = 0.5 * tf.reduce_sum(tf.square(tf.divide((inputs - outputs), self.wgt))) #changed tf.divide
        kl_loss             = -0.5 * tf.reduce_sum(1 + (z_log_var) -
                               tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss


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

def l_mod2(inpt = 27): # M3, no LR
    
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


def deviance(y, p):
    eps = 1e-8 # originally 10-16 but was instable
    y = tf.cast(y, dtype=tf.float32)
    #p = tf.convert_to_tensor(p, dtype=tf.float32)
    p = tf.clip_by_value(p, eps, 1 - eps)  # Clip predictions to avoid log of zero
    return -2 * tf.reduce_sum(y * tf.math.log((p + eps) / (1 - p + eps)) + tf.math.log(1 - p + eps))



def deviance2(y, p):
    eps = 1e-8  # Small epsilon to avoid log(0)
    
    # Convert y and p to numpy arrays if they are not already
    y = np.array(y, dtype=np.float32)
    p = np.array(p, dtype=np.float32)
    
    # Clip predictions to avoid log of zero
    p = np.clip(p, eps, 1 - eps)
    
    # Compute deviance
    deviance_value = -2 * np.sum(y * np.log((p + eps) / (1 - p + eps)) + np.log(1 - p + eps))
    
    return deviance_value

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

c_weight2 = {0 : 0.6834631613112626, 1 :1.862671384343211}
#classes, count = np.unique(Y, return_counts=True)
#class_freq = count/len(Y)
#compute inverse of class frequencies
#class_w1 = dict(enumerate(1 / class_freq))
#del(classes, count)
file_path = "C:/Users/User/Documents/Data Science/LDATS2310 DS for insu and finance/"
#%% 1 Load data and  Preprocessing

car_insu = pd.read_excel(file_path + "01Data/Car Insurance Claim.xlsx",index_col="ID")
data = deepcopy(car_insu)
print(data.shape)



to_keep = ['Age',"Single Parent?","Marital Status","Gender","Education","Occupation",
           "Car Use","Car Type","City Population"]
data = data[to_keep]

# Categorize age
data['Age'].describe()


# =============================================================================
# plt.figure(figsize=(10, 6))
# plt.hist(car_insu['Age'], bins=30, color='skyblue', edgecolor='black')
# plt.title('Distribution of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# =============================================================================

# Categories of different size based on what I think is the best percentile
data['Cat_age'] = pd.cut(data["Age"],bins= [0,30,40,50,82],labels=["0-30","30-40","40-50","50-.."])
data['Cat_age'].value_counts()
data.drop("Age", axis = 1,inplace = True)

# get 01 binary var
data['Urb'] = (data['City Population'] == "Urban").astype("int64")
data["Car Use"] = (data['Car Use'] == "Private").astype("int64")
data["Gender"] = (data['Gender'] == "M").astype("int64")
data["Marital Status"] = (data['Marital Status'] == "Yes").astype("int64")
data["Sing. Par"] = (data['Single Parent?'] == "Yes").astype("int64")

# multilevel categorical variable
to_categorical = ["Education","Occupation","Car Type"]
data[to_categorical] = data[to_categorical].astype('category')

data.drop(["City Population","Single Parent?"],axis = 1, inplace = True)

data_table = deepcopy(data)
#%% 2 Check for correlated variable

def cramerV(label,x):
    confusion_matrix = pd.crosstab(label, x)
    chi2 = chi2_contingency(confusion_matrix)[0] # get the test statistic's value
    n = confusion_matrix.sum().sum() # calculates the total number of individuals
    r,k = confusion_matrix.shape # get the levels of both categorical variables (rows and columns)
    phi2 = chi2/n # phi2 coef
    #correction applied to avoid a bias
    phi2corr = max(0,phi2-((k-1)*(r-1))/(n-1)) # phi2 coef with correction applied
    rcorr = r - ((r - 1) ** 2) / ( n - 1 ) # correction
    kcorr = k - ((k - 1) ** 2) / ( n - 1 ) # correction
    try:
        if min((kcorr - 1),(rcorr - 1)) == 0:
            #warnings.warn("Unable to calculate Cramer's V using bias correction. Consider not using bias correction",RuntimeWarning)
            v = 0
            print("If condition Met: ",v)
        else:
            v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
            print("Else condition Met: ",v)
    except:
        print("inside error")
        v = 0
    return v

def plot_cramer(df):
    cramer = pd.DataFrame(index=df.columns,columns=df.columns)
    for column_of_interest in df.columns:
        try:
            temp = {}

            columns = df.columns
            for j in range(0,len(columns)):
                v = cramerV(df[column_of_interest],df[columns[j]])
                cramer.loc[column_of_interest,columns[j]] = v
                if (column_of_interest==columns[j]):
                    pass
                else:
                    temp[columns[j]] = v
            cramer.fillna(value=np.nan,inplace=True)
        except:
            print('Dropping row:',column_of_interest)
            pass
    plt.figure(figsize=(7,7))
    sns.heatmap(cramer,annot=True,fmt='.2f')

    plt.title("Cross Correlation plot on Dataframe with Cramer's Correlation Values")
    plt.show()
    
plot_cramer(data)

cramerV(data['Education'],data["Occupation"]) #0.64
cramerV(data["Car Type"], data['Gender']) # 0.71
cramerV(data["Car Type"], data['Car Use']) # 0.47
cramerV(data["Occupation"], data['Car Use']) # 0.57
cramerV(data["Marital Status"], data['Sing. Par']) # 0.48


#%% dummification

data = pd.get_dummies(data, columns = ["Education","Car Type","Cat_age","Occupation"],drop_first= True)
data.iloc[:,5:] = data.iloc[:,5:].astype("float64")

#%% 4 Analyze the variability and scale variable

variance = data.var()
std = data.std()
a = variance > 2
to_scale = list(variance[a].index)
if not to_scale:
    print('EMPTY')

# check
del(a,variance,std)

#%% 5 CREATING training set

X = np.asarray(data)

# creating train and tests sets
x_train, x_test= train_test_split(X, test_size=0.2,random_state=4243)


#%% needed for Hyperband
'''
#to launch tensorboard # ???
import subprocess
log_dir = "C:/Users/User/Documents/Data Science/LINFO2262 ML/A5_Competition/DNN/logs/fit2/"
command = ["tensorboard","--logdir",log_dir]
subprocess.run(command, check = True)
'''

nc = x_train.shape[1]
std_data = np.std(x_train,axis = 0)

def build_VAE1(hp): 
    # try implement the encode and decoder in two different architecture
    
    latent_dim = hp.Int('latent_dim',min_value = 4, max_value = 24, step = 2)
    enco_inpt = Input(shape=(nc,))
    deco_inpt = Input(shape=(latent_dim,))
    
    hle = enco_inpt
    hld = deco_inpt
    
    ###### ENCODER PART
    #--- Loop to find optimal number of layer
    for i in range(hp.Int("n_layersE",min_value = 1, max_value = 5, step = 1)):
        #--- Look for optimal n_neurons
        n_neurons = hp.Int(f"n_neu_E{i}", min_value = 16, max_value = 256,step = 16)
        acti = hp.Choice(f'activation_E{i}',["selu","relu","tanh","softplus"])
        #--- If it's the first layer
        if i == 1:
            #Hidden Layer Encoder
            hle = Dense(units = n_neurons, activation = acti)(enco_inpt)
        else:
            hle = Dense(units = n_neurons, activation = acti )(hle)
            
    
    # latent_dim 
    z_mean = Dense(latent_dim,name="z_mean")(hle)
    z_log_var = Dense(latent_dim,name="z_log_Var")(hle)
    
    
    #### DECODER
    for i in range(hp.Int("n_layersD",min_value = 1, max_value = 5, step = 1)):
        #--- Look for optimal n_neurons
        n_neurons = hp.Int(f"n_neu_D{i}", min_value = 16, max_value = 256,step = 16)
        acti = hp.Choice(f'activation_D{i}',["selu","relu","tanh","softplus"])
        #--- If it's the first layer
        if i == 1:
            #Hidden Layer Decoder
            hld = Dense(units = n_neurons, activation = acti)(deco_inpt)
        else:
            hld = Dense(units = n_neurons, activation = acti )(hld)
    
    
    out_deco = hp.Choice('acti_dec', ['sigmoid','tanh','softplus','selu'])
    decoder_out = Dense(nc, activation = out_deco)(hld)
    
    #create var enco and deco
    var_encoder = keras.Model(inputs = [enco_inpt], outputs = [z_mean,z_log_var],name = "encoder")
    var_decoder = keras.Model(inputs = [deco_inpt], outputs = [decoder_out],name = "decoder")
    
    vae = VAE(var_encoder,var_decoder,std_data)
    #lr = hp.Choice('lear_r',[0.01,0.001])
    optim = hp.Choice("optim", ['adam',"rmsprop"])
    vae.compile(optimizer=optim)
    
    return vae


def build_VAE2(hp):  # same version but with dropout layers
   
    
    latent_dim = hp.Int('latent_dim',min_value = 2, max_value = 10, step = 2) # itinialy higher
    enco_inpt = Input(shape=(nc,))
    deco_inpt = Input(shape=(latent_dim,))
    
    hle = enco_inpt
    hld = deco_inpt
    
    ###### ENCODER PART
    #--- Loop to find optimal number of layer
    for i in range(hp.Int("n_layersE",min_value = 1, max_value = 5, step = 1)):
        #--- Look for optimal n_neurons
        n_neurons = hp.Int(f"n_neu_E{i}", min_value = 32, max_value = 256,step = 32) # # changement, min value can be 0
        acti = hp.Choice(f'activation_E{i}',["selu","relu","tanh","softplus"])
        #--- If it's the first layer
        if i == 1:
            #Hidden Layer Encoder
            hle = Dense(units = n_neurons, activation = acti)(enco_inpt)
        else:
            hle = Dense(units = n_neurons, activation = acti )(hle)
            
        if hp.Boolean(f'dropoutE_{i}') and n_neurons > 0:
            r = hp.Float(f'rate_E{i}',min_value = 0.1, max_value = 0.5, step = 0.2)
            hle = Dropout(rate = r) (hle)
            
    # latent_dim 
    z_mean = Dense(latent_dim,name="z_mean")(hle)
    z_log_var = Dense(latent_dim,name="z_log_Var")(hle)
    
    
    #### DECODER
    for i in range(hp.Int("n_layersD",min_value = 1, max_value = 5, step = 1)):
        #--- Look for optimal n_neurons
        n_neurons = hp.Int(f"n_neu_D{i}", min_value = 32, max_value = 256,step = 32) # changement, min value can be 0
        acti = hp.Choice(f'activation_D{i}',["selu","relu","tanh","softplus"])
        #--- If it's the first layer
        if i == 1:
            #Hidden Layer Decoder
            hld = Dense(units = n_neurons, activation = acti)(deco_inpt)
        else:
            hld = Dense(units = n_neurons, activation = acti )(hld)
            
        if hp.Boolean(f'dropoutD_{i}') and n_neurons > 0:
            r = hp.Float(f'rate_D{i}',min_value = 0.1, max_value = 0.5, step = 0.2) # reduce steps
            hle = Dropout(rate = r) (hle)
    
    out_deco = hp.Choice('acti_dec', ['sigmoid','tanh',])
    decoder_out = Dense(nc, activation = out_deco)(hld)
    
    #create var enco and deco
    var_encoder = keras.Model(inputs = [enco_inpt], outputs = [z_mean,z_log_var],name = "encoder")
    var_decoder = keras.Model(inputs = [deco_inpt], outputs = [decoder_out],name = "decoder")
    
    vae = VAE(var_encoder,var_decoder,std_data)
    #lr = hp.Choice('lear_r',[0.01,0.001])
    optim = hp.Choice("optim", ['adam',"rmsprop"])
    vae.compile(optimizer=optim)
    
    return vae

#%% Hyperband tuner
# not needed to run, model set up later
tunerVAE1 = Hyperband(
    build_VAE1,
    objective ='val_loss',
    max_epochs = 100,
    hyperband_iterations = 3,
    directory = file_path + "New_logs/VAE",
    project_name ="model1",
    max_consecutive_failed_trials= 5)


log_dir = "C:/Users/User/Documents/Data Science/LDATS2310 DS for insu and finance/New_logs/VAE/model1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)

# model 1
tunerVAE1.search(x_train,x_train,
                 validation_data = (x_test,x_test),
              epochs=100, 
             batch_size= 512, #arbitraly chosen
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5,restore_best_weights = True),
                        #lr_scheduler,
                        tensorboard_callback])

best_hps_val1 = tunerVAE1.get_best_hyperparameters(num_trials=1)[0].values
best_hps1 = tunerVAE1.get_best_hyperparameters(num_trials=1)[0]
modelVAE1 = tunerVAE1.hypermodel.build(best_hps1)
modelVAE1.summary()

# =============================================================================
# for layer in modelVAE1.layers:
#     print('-'*40)
#     print(layer.get_config(),"\n")
# 
# decoder = modelVAE1.get_layer("decoder")
# decoder.summary()
# 
# =============================================================================

#%% Hyperband 2

tunerVAE2 = Hyperband(
    build_VAE2,
    objective ='val_loss',
    max_epochs = 50,
    hyperband_iterations = 3,
    directory = "C:/Users/User/Documents/Data Science/LDATS2310 DS for insu and finance/New_logs/VAE",
    project_name ="model5",
    max_consecutive_failed_trials= 5)


log_dir = "C:/Users/User/Documents/Data Science/LDATS2310 DS for insu and finance/New_logs/VAE/model5/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)

# model 2
tunerVAE2.search(x_train,x_train,
                 validation_data = (x_test,x_test),
              epochs=50, 
             batch_size= len(x_train), #arbitraly chosen
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5,restore_best_weights = True),
                        #lr_scheduler,
                        tensorboard_callback])

best_hps_val2 = tunerVAE2.get_best_hyperparameters(num_trials=1)[0].values
best_hps2 = tunerVAE2.get_best_hyperparameters(num_trials=1)[0]
modelVAE2 = tunerVAE2.hypermodel.build(best_hps2)
modelVAE2.get_layer("encoder").summary()
modelVAE2.get_layer("decoder").summary()


#%% building model 1


# =============================================================================
# decoder = modelVAE1.get_layer("decoder")
# encoder = modelVAE1.get_layer("encoder")
# decoder.summary()
# encoder.summary()
# a = tunerVAE1.get_best_hyperparameters()[0].values
# b = tunerVAE1.get_best_models()
# 
# =============================================================================
def build_model1():
    l_dim = 16
    # encoder
    enco_inpt = Input(shape=(nc,))
    # Hidden Layer Encoder
    hle = Dense(224,activation = 'tanh')(enco_inpt)
    
    z_mean = Dense(l_dim, name = 'z_mean')(hle)
    z_log_var = Dense(l_dim,name = 'z_log_var')(hle)
    
    # Decoder
    deco_inpt = Input(shape = (l_dim,))
    hld = Dense(units = 208,activation = 'selu')(deco_inpt)
    hld = Dense(units =128,activation = 'relu')(hld)
    hld = Dense(units =192,activation = 'relu')(hld)
    hld = Dense(units  = 80,activation = 'softplus')(hld)
    
    decoder_out = Dense(units = nc,activation = 'tanh')(hld)
    
    #create var enco and deco
    var_encoder = keras.Model(inputs = [enco_inpt], outputs = [z_mean,z_log_var],name = "encoder")
    var_decoder = keras.Model(inputs = [deco_inpt], outputs = [decoder_out],name = "decoder")
    
    # create VAE object
    vae = VAE(var_encoder,var_decoder,std_data)
    
    #compile it 
    vae.compile(optimizer = "adam")
    
    return vae

#%% build model 2
# =============================================================================
# modelVAE2.get_layer("encoder").summary()
# modelVAE2.get_layer("decoder").summary()
# best_hps_val2
# 
# for layer in modelVAE2.layers:
#     print('-'*40)
#     print(layer.get_config(),"\n")
# =============================================================================
l_dim2 = 8
def build_model2():
    l_dim = 8
    
    #--- enco
    enco_inpt = Input(shape =(nc,))
    
    #hidden Layer enco
    hle = Dense(units = 256,activation = 'tanh')(enco_inpt)
    hle = Dropout(rate =0.3 )(hle)
    
    # latent lvl
    z_mean = Dense(l_dim, name = 'z_mean')(hle)
    z_log_var = Dense(l_dim, name ='z_var')(hle)
    
    #--- Decoder
    deco_inpt = Input(shape = (l_dim,))
    hld = Dense(units = 32,activation = 'tanh')(deco_inpt)
    hld = Dense(units = 224,activation = 'tanh')(hld)
    hld = Dense(units = 256,activation = 'softplus')(hld)
    
    decoder_out = Dense(units =nc, activation ='sigmoid')(hld)
    
    #create var enco and deco
    var_encoder = keras.Model(inputs = [enco_inpt], outputs = [z_mean,z_log_var],name = "encoder")
    var_decoder = keras.Model(inputs = [deco_inpt], outputs = [decoder_out],name = "decoder")
    
    # create VAE object
    vae = VAE(var_encoder,var_decoder,std_data)
    
    #compile it 
    vae.compile(optimizer = 'rmsprop')
    
    return vae
    
#%% build model12
# =============================================================================
# modelVAE1.get_layer("encoder").summary()
# modelVAE1.get_layer("decoder").summary()
# best_hps_val1
# 
# for layer in modelVAE1.layers:
#      print('-'*40)
#      print(layer.get_config(),"\n")
# 
# =============================================================================
l_dim3 = 18
def build_model3():
    l_dim = 18
    
    #--- enco
    enco_inpt = Input(shape =(nc,))
    
    #hidden Layer enco
    hle = Dense(units = 240,activation ='selu')(enco_inpt)
    hle = Dense(units = 160,activation ='tanh')(hle)
    hle = Dense(units = 160,activation = 'tanh')(hle)
    
    z_mean = Dense(l_dim, name = 'z_mean')(hle)
    z_log_var = Dense(l_dim, name ='z_var')(hle)  
    
    #--- Decoder
    deco_inpt = Input(shape = (l_dim,))
    hld = Dense(units = 48,activation = 'softplus')(deco_inpt)
    hld = Dense(units = 48,activation = 'tanh')(hld)
    hld = Dense(units = 208,activation = 'tanh')(hld)
    hld = Dense(units = 128,activation = 'relu')(hld)
    
    decoder_out = Dense(units =nc,activation = 'sigmoid' )(hld)
    
    #create var enco and deco
    var_encoder = keras.Model(inputs = [enco_inpt], outputs = [z_mean,z_log_var],name = "encoder")
    var_decoder = keras.Model(inputs = [deco_inpt], outputs = [decoder_out],name = "decoder")
    
    # create VAE object
    vae = VAE(var_encoder,var_decoder,std_data)
    
    #compile it 
    vae.compile(optimizer = 'adam')
    
    return vae

#%% build model 3
# =============================================================================
# modelVAE2.get_layer("encoder").summary()
# modelVAE2.get_layer("decoder").summary()
# best_hps_val2
# 
# for layer in modelVAE2.layers:
#      print('-'*40)
#      print(layer.get_config(),"\n")
# =============================================================================
#l_dim12 = 12
def build_model4():
    l_dim = 12
    
    #--- enco
    enco_inpt = Input(shape =(nc,))
    
    #hidden Layer enco
    hle = Dense(units = 192,activation ='selu')(enco_inpt)
    hle = Dropout(rate = 0.5 )(hle)
    hle = Dense(units = 256,activation = 'selu')(hle)
    
    z_mean = Dense(l_dim, name = 'z_mean')(hle)
    z_log_var = Dense(l_dim, name ='z_var')(hle)  
    
    #--- Decoder
    deco_inpt = Input(shape = (l_dim,))
    hld = Dense(units = 256,activation ='selu' )(deco_inpt)
    hld = Dense(units = 128,activation = 'selu')(hld)
    hld = Dense(units = 128,activation = 'softplus')(hld)

    
    decoder_out = Dense(units =nc,activation = 'sigmoid' )(hld)
    
    #create var enco and deco
    var_encoder = keras.Model(inputs = [enco_inpt], outputs = [z_mean,z_log_var],name = "encoder")
    var_decoder = keras.Model(inputs = [deco_inpt], outputs = [decoder_out],name = "decoder")
    
    # create VAE object
    vae = VAE(var_encoder,var_decoder,std_data)
    
    #compile it 
    vae.compile(optimizer = 'adam')
    
    return vae

def rec_loss(inputs, outputs, wgt):
    diff = inputs - outputs
    # sum for all col
    a = diff/wgt
    
    # get the weighted squarred of a
    
    w_squarred = np.square(a) # we have the square of the weighted difference
    
    # sum everything twice cuz it wont sum accross all elements
    total_loss = np.sum(w_squarred)
    return np.sum(total_loss)

def train_mod(n_iter,latent_dim,build_model,bs,start_from = 150):
    w = np.std(x_test, axis = 0)
    n_p = len(x_test)
    ES = EarlyStopping(patience =5, monitor ='val_loss',verbose = 0, mode = 'min', restore_best_weights = True, start_from_epoch = start_from)
    losses = []
    for i in range(n_iter):
        mod = build_model
        mod.fit(x_train,x_train,
                verbose = 0,
                validation_data =(x_test,x_test),
                epochs = 250, 
                batch_size = bs,
                callbacks = [ES]
            )
        # deco part
        deco = mod.get_layer('decoder')
        
        # enco part
        z_mean, z_log_var = mod.encoder.predict(x_test)
        
        # to introduce noise/ randomness
        eps = np.random.normal(size = z_mean.shape)
        
        # scale and shift the noise following the learned dist
        # np.exp(0.5 * z_log_var) is the std
        z = z_mean + np.exp(0.5 * z_log_var) * eps
        
        # generate the new data
        new_d = deco(z).numpy()
        
        losses.append(rec_loss(x_test, new_d, w))
        print(f'run {i} over \n')
        
    #print('ITS FUCKING OVER BOIIII')
    return losses, np.mean(losses)

#%% test the model1
earl_stop1 = EarlyStopping(patience =5, monitor ='val_loss',verbose = 1, mode = 'min', restore_best_weights = True, start_from_epoch = 120) #baseline = 2000,

model1 = build_model4()
history1 = model1.fit(x_train,x_train,
                      validation_data =(x_test,x_test),
                      epochs = 250, batch_size = len(x_train),callbacks = [earl_stop1])


perf_mod1 = train_mod(10,l_dim2,build_model2(),len(x_train),0)
perf_mod1[1]

# =============================================================================
# #analyze the results
# deco1 = model1.get_layer('decoder')
# n_points = len(x_test)
# l_dim1 = 12
# to_generate = tf.random.normal(shape=[n_points,l_dim1])
# new_data = deco1(to_generate)
# weights = np.std(x_test,axis = 0)
# rec_loss(x_test,new_data,weights) # 40 512.71 // 40224.02
# 
# 
# enco1 = model1.get_layer('encoder')
# zm, zvar = model1.encoder.predict(x_test)
# 
# epsilon = np.random.normal(size=zm.shape)
# z = zm + np.exp(0.5 * zvar) * epsilon
# 
# test = deco1(z).numpy()
# rec_loss(x_test, test, weights) # 42715.95 //  37451.43550767362 //
# =============================================================================

#%% plots1
#--- Plot the training and validation loss
plt.figure(figsize=(10, 6))

# Plot training loss
plt.plot(history1.history['loss'], label='Training Loss')

# If you have validation data, plot validation loss
if 'val_loss' in history1.history:
    plt.plot(history1.history['val_loss'], label='Validation Loss')

plt.title('Model1 Loss Over Epochs (BS =full batch)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#%% testing model 2 to find optimal
earl_stop2 = EarlyStopping(patience =5, monitor ='val_loss',verbose = 1, mode = 'min', restore_best_weights = True, baseline = 2000,start_from_epoch = 120)

model2 = build_model2()
history2 = model2.fit(x_train,x_train,
                      validation_data = (x_test,x_test),
                      epochs = 250,
                      batch_size =len(x_train),
                      callbacks = [earl_stop2])

# Plot the training and validation loss
plt.figure(figsize=(10, 6))

# Plot training loss
plt.plot(history2.history['loss'], label='Training Loss')

# If you have validation data, plot validation loss

if 'val_loss' in history2.history:
    plt.plot(history2.history['val_loss'], label='Validation Loss')

plt.title('Model2 Loss Over Epochs (BS =128)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



#%%  Testing model
l_dim = [16, 8,18,12]
perf_mod1 = train_mod(20,l_dim[0],build_model1(),len(x_train),120)
perf_mod2 = train_mod(20,l_dim[1],build_model2(),len(x_train),120)
perf_mod3 = train_mod(20,l_dim[2],build_model3(),len(x_train),120)
perf_mod4 = train_mod(20,l_dim[3],build_model4(),len(x_train),120)

print(round(perf_mod1[1],2))
print(round(perf_mod2[1],2))
print(round(perf_mod3[1],2))
print(round(perf_mod4[1],2))

print(round(np.std(perf_mod1[0]),2))
print(round(np.std(perf_mod2[0]),2))
print(round(np.std(perf_mod3[0]),2))
print(round(np.std(perf_mod4[0]),2))

np.min(perf_mod1[0]) # 40 306.85
np.max(perf_mod1[0]) # 58 817.707
np.std(perf_mod1[0])

perf = train_mod(30,l_dim[3],build_model4(),len(x_train),120)
print(round(perf[1],2))
print(round(np.std(perf[0]),2))

#%% KMEANS PART
from sklearn.cluster import KMeans
ESK = EarlyStopping(patience =5, monitor ='val_loss',verbose = 1, mode = 'min', restore_best_weights = True, start_from_epoch = 120)

#-- MODEL 4
modK = build_model4()
modK.fit(X,X,
         epochs = 250,
         batch_size =len(X),
         callbacks = [ESK])



# get X compressed
encoK = modK.get_layer('encoder')
X_comp4,_ = encoK.predict(X)

#-- MODEL 2
modK2 = build_model2()
modK2.fit(X,X,
         epochs = 250,
         batch_size =len(X),
         callbacks = [ESK])
encoK2 = modK2.get_layer('encoder')
X_comp2,_ = encoK2.predict(X)

#%% compare the two models

n_clust = 10
kmeans_model2 = KMeans(n_clusters=n_clust, init='k-means++', n_init=20, max_iter=300, random_state=4243)
kmeans_model2.fit(X_comp2)
print(kmeans_model2.inertia_)

kmeans_model4 = KMeans(n_clusters=n_clust, init='k-means++', n_init=20, max_iter=300, random_state=4243)
kmeans_model4.fit(X_comp4)
print(kmeans_model4.inertia_)

in_M2 = []
in_M4 = []
clusters = [5, 6, 7, 8, 9, 10]
for c in clusters:
    kmeans_model2 = KMeans(n_clusters=c, init='k-means++', n_init=20, max_iter=300, random_state=4243)
    kmeans_model2.fit(X_comp2)
    in_M2.append(kmeans_model2.inertia_)
    
    kmeans_model4 = KMeans(n_clusters=c, init='k-means++', n_init=20, max_iter=300, random_state=4243)
    kmeans_model4.fit(X_comp4)
    in_M4.append(kmeans_model4.inertia_)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(clusters, in_M2, marker='o', label='Model 2 Inertia')
plt.plot(clusters, in_M4, marker='s', label='Model 4 Inertia')

# Adding titles and labels
plt.title('Model Inertia vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.legend()

# Show plot
plt.grid(True)
plt.show()

# MODEL 2 FOR SURE
#%% Optimal number of cluster if inertia == deviance => MOD2

# Range of cluster numbers to try
k_values = range(1, 15)
inertias = []

# Perform K-means for each k and calculate inertia
for k in k_values:
    kmeans = KMeans(init = 'k-means++',
                n_clusters = k,
                n_init = 20,
                max_iter = 300,
                random_state=4243)
    kmeans.fit(X_comp2)
    inertias.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method For Optimal k M2')
plt.grid(True)
plt.show()
#between 8 and 10 there 
# 7 seems fine

from sklearn.metrics import silhouette_score

sil_scores = []

for k in range(2, 11):  # Silhouette score is only defined for k >= 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_comp2)
    sil_score = silhouette_score(X_comp2, labels)
    sil_scores.append(sil_score)

# Plotting the Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k M2 ')
plt.grid(True)
plt.show()

#%% identify profile 
n_cluster = 8

def make_tab2(km, n_clust, df, DS):
    km.fit(df)
    print("Inertia:", km.inertia_)
    
    # Indicates to which class each point of the DS correspond
    X_lab_Clust = km.labels_  
    
    # Creating the table
    names = DS.columns.tolist()
    #tab = np.zeros(shape=(n_clust, len(names) + 1))  # +1 for 'prop_ind' column
    #tab = pd.DataFrame(data=tab, columns=names + ['prop_ind'])
    tab = pd.DataFrame(np.zeros(shape=(n_clust, len(names) + 1)), columns=names + ['prop_ind'], dtype=object)
    tab = tab.astype(float)  # Ensure the dataframe is of type float
    
    # Filling the dataframe
    for k in range(n_clust):
        # Get the index of the data points that belongs to the current class
        idx = (X_lab_Clust == k)
        
        # Computing the proportion of individuals per cluster
        tab.at[k, "prop_ind"] = round((sum(idx) / len(DS)) * 100, 2)
        
        # Filling in the most common value for each column in the cluster
        for col in names:  # Iterate over the original columns only
            tab.at[k, col] = Counter(DS[col][idx]).most_common(1)[0][0]
    
    return tab


km1 = KMeans(init = 'k-means++',
            n_clusters = n_cluster,
            n_init = 20,
            max_iter = 300,
            random_state=4243)

table2 = make_tab2(km1, n_cluster, X_comp2,data_table) #work on table 2 cuz model 2 has the smallest loss
table2.to_csv('kmeans_table.csv',sep=",")
#%% Analyze the table
car_insu['Marital Status'].value_counts() # +- 0.625 married
car_insu['City Population'].value_counts() # 80 % Urb
car_insu['Single Parent?'].value_counts() # 86 %

import dataframe_image as dfi
dfi.export(table2,"Profile table.png")
