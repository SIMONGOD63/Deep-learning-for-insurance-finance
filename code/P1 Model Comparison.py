#%% 0 packages
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score 
from imblearn.over_sampling import SMOTE
from statistics import variance
from copy import deepcopy
from collections import Counter
from scikeras.wrappers import KerasRegressor
from adjustText import adjust_text
file_path = "C:/Users/User/Documents/Data Science/LDATS2310 DS for insu and finance/"

#--- Model found with hyperband tuner
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

#c_weight2 = {0 : 0.6834631613112626, 1 :1.862671384343211}
os.chdir(file_path)
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

# data leakage through scaling !!

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

#%% Testing the saved models
nc = x_train.shape[1]
last_mod2 = l_mod2(nc)
last_mod1 = l_mod1(nc)
last_mod3 = l_mod3(nc)
l_test = l_mod3(nc)
last_mod4 = l_mod4(nc)

last_mod2.load_weights('Mod_weights/mod_last2.weights.h5')
last_mod1.load_weights('Mod_weights/mod_last1.weights.h5')
last_mod3.load_weights('Mod_weights/mod_last3.weights.h5')
last_mod4.load_weights('Mod_weights/mod_last4.weights.h5')
l_test.load_weights('Mod_weights/mod44.weights.h5')

models = [last_mod1,last_mod2,last_mod3,last_mod4,l_test]

for i in range(len(models )):
    
    pred = (models[i].predict(x_test,verbose = 0) > 0.5).astype(int)
    acc =accuracy_score(y_test, pred) 
    bcr = BCR(y_test,pred) 
    dev = deviance2(y_test, models[i].predict(x_test,verbose = 0)) 
    
    print(f'M{i + 1} : loss : {dev}, acc : {acc} , bcr : {bcr} \n')

del(pred,acc,bcr,dev)
#%% Inspect the best model
nc = x_train.shape[1]
last_mod2 = l_mod2(nc)
last_mod2.load_weights('Mod_weights/mod_last2.weights.h5')
pred = (last_mod2.predict(x_test) > 0.5).astype(int)
accuracy_score(y_test, pred) #.7454
BCR(y_test,pred)# .7339

#%% Loading the model

#---Creating unsureness label
nc = x_train.shape[1]
last = l_mod2(nc)
last.load_weights('Mod_weights/mod_last2.weights.h5')

#%% Creating WDS
pred = (last.predict(x_test) > 0.5).astype(int)
output = last.predict(x_test)
lab = np.zeros_like(pred)
unsure = 0.07
# label the unsure prediction of my model (close to 0.5)
lab[ (0.5 <= output) & (output < 0.5 + unsure) ] = 2
lab[ (output> 0.5 - unsure) & (output <= 0.5)] = -2

# create a Working DataSet
wds = deepcopy(x_test)
wds = pd.DataFrame(wds)

# Add column's name
wds.columns = list(data.drop("Crash",axis = 1).columns)
wds['Crash'] = y_test
wds['Predict.'] = pred
wds['Sure'] = lab
wds['output'] = output

del(pred,output,lab,unsure)

#wds.to_csv('wds_sure007.csv', index=False)
#wds.to_pickle('wds_sure007.pkl')

#%% load the model
#wds = pd.read_pickle('wds_sure007.pkl')
#%%-- Needed function and load WDS

def get_info(col_name,data):
    '''    
    Parameters
    ----------
    col_name : str
        column name.

    Returns
    -------
    out : pd.DataFrame
        Return a dataFrame with the observed and prediction of Crash within 
        the given category set by the column name.

    '''
    # if I pass a cond as col_name to get the suppressed category by the dummification
    if type(col_name) is not str:
        
        print("NOT STR BUT COND")
        ds = data[col_name]
        
    else :
        ds = data[data[col_name] == 1]
        
    stock = list()
    
    # create a subset where the given category is active
    
    acc = round(accuracy_score(ds['Crash'],ds['Predict.']) ,4)
    obs = dict(Counter(ds['Crash']))
    pred = dict(Counter(ds['Predict.']))
    c_obs = round(obs[1] /(obs[1] + obs[0]),4)*100
    c_pred = round(pred[1] /(pred[1] + pred[0]),4)*100
    diff1 = round(c_pred - c_obs,2) # pos => mod predict more
    sureness = round((sum(abs(ds['Sure']))/2)/len(ds),4)*100 # represent the proportion number of points for which model is "unsure"
    stock.append((col_name,acc,
                  c_obs,c_pred,diff1,sureness,len(ds)))
    out = pd.DataFrame(stock,columns= ['Col name',"Accuracy",
                                   "Crash Obs (%)","Crash Pred(%)",
                                   "Crash P - O (%)","Unsureness","Samp. Size"])

    return out

#mast = get_info("Masters")
# check porportion of the suppressed category
# it works as expected.

dst = wds[(wds["High School"] == 0) & (wds["Masters"] == 0) & (wds["PhD"] == 0)]
Counter(dst['Crash'])[1]/(Counter(dst['Crash'])[0] + Counter(dst['Crash'])[1])
un = get_info((wds["High School"] == 0) & (wds["Masters"] == 0) & (wds["PhD"] == 0),wds)

dst2 = wds[(wds["Panel Truck"] == 0) & (wds["Pickup"] == 0) & (wds["SUV"] == 0) & (wds['Sports Car'] == 0) & (wds["Van"] == 0)]
Counter(dst2['Crash'])[1]/(Counter(dst2['Crash'])[0] + Counter(dst2['Crash'])[1])
deux = get_info((wds["Panel Truck"] == 0) & (wds["Pickup"] == 0) & (wds["SUV"] == 0) & (wds['Sports Car'] == 0) & (wds["Van"] == 0),wds)


def make_table(list_col,data):
    '''
    The function use get_info and combined all categories we want.

    Parameters
    ----------
    list_col : list
        list of column name to group together.

    Returns
    -------
    stock : pd.DataFrame
        combination of the dataFrame.

    '''
    # this function should merge all the output i get from get_info
    stock = None
    
    for col in list_col:
        stock = pd.concat((stock,get_info(col,data)), axis = 0)
        
    return stock  

#test = wds[wds['Masters'] == 0 & wds['PhD'] == 0]
del(dst,dst2,un,deux)

#%% testing if the functions work as expected
# =============================================================================
# wds_old = pd.read_pickle('wds_datap.pkl')
# 
# to_inves = ["Gender","Car Use","Urb","Lics R","Single Parent?","Marital Status"]
# to_inves = [*to_inves,*list(e_dum.columns),*list(c_dum.columns)]
# pre_a = make_table(to_inves,wds_old)
# bbb = get_info((wds_old["High School"] == 0) & (wds_old["Masters"] == 0) & (wds_old["PhD"] == 0),wds_old)
# ccc = get_info((wds_old["Panel Truck"] == 0) & (wds_old["Pickup"] == 0) & (wds_old["SUV"] == 0) & (wds['Sports Car'] == 0) & (wds_old["Van"] == 0),wds_old)
# analyze_old = pd.concat((pre_a,bbb,ccc),axis = 0)
# analyze_old.iloc[14,0] = "Bachelors"
# analyze_old.iloc[15,0] = "Minivan"
# =============================================================================

#%% Creating the table to analyze
to_inves = ["Gender","Car Use","Urb","Lics R","Sing. Par","Marital Status"]
to_inves = [*to_inves,*list(e_dum.columns),*list(c_dum.columns)]
pre_a = make_table(to_inves,wds)
bbb = get_info((wds["High School"] == 0) & (wds["Masters"] == 0) & (wds["PhD"] == 0),wds)
ccc = get_info((wds["Panel Truck"] == 0) & (wds["Pickup"] == 0) & (wds["SUV"] == 0) & (wds['Sports Car'] == 0) & (wds["Van"] == 0),wds)
analyze = pd.concat((pre_a,bbb,ccc),axis = 0)
analyze.iloc[14,0] = "Bachelors"
analyze.iloc[15,0] = "Minivan"

del(pre_a,bbb,ccc,to_inves,e_dum,c_dum)
#%% Analyze my results

# mean diff
np.mean(analyze['D. Crash: P - O (%)']) # 7.815

# Observed proportion of Crash
Counter(data['Crash'])[1]/(Counter(data['Crash'])[1] + Counter(data['Crash'])[0]) # 26.84%

# Predicted proportion of Crash
Counter(wds['Predict.'])[1]/(Counter(wds['Predict.'])[1] + Counter(wds['Predict.'])[0]) # 0.3703 # previously 36.02% //.3341 ??

# M1 => 0.37032640949554896
#M2 =>  0.37032640949554896

#%% using math plotlib


# Assuming 'analyze' is your DataFrame
x = analyze['Crash Obs (%)'].tolist()
y = analyze['Crash Pred(%)'].tolist()
sizes = analyze['Samp. Size']
sizes2 = analyze['Unsureness']
labels = list(analyze['Col name'])

# Creating the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot with color scale
scatter = ax.scatter(x, y, c=sizes, cmap='coolwarm', s=100, edgecolor='k')

# Adding the y = x line
ax.plot([0, 100], [0, 100], linestyle='--', color='black')

#Adding its legend
ax.text(x= 5.5, y=  2.5 , s='y = x', fontsize=12, color='black', ha='right',rotation = 45)


# Adding dashed lines for means
ax.axhline(y=37.03, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=26.84, color='gray', linestyle='--', alpha=0.3)

# Adding annotations for the means
ax.text(2, 33.8, 'Pred. Crash mean', fontsize=12, alpha=0.6)
ax.text(27.5, 1, 'Obs. Crash mean', fontsize=12, alpha=0.6)
plt.text(-0.5, 37.03, '37.03', va='center', ha='right', color='grey', fontsize=8)
plt.text(26.84, -1.5, '26.84', va='center', ha='center', color='grey', fontsize=8)


# Adding text labels
texts = []
for i, label in enumerate(labels):
    texts.append(ax.text(x[i], y[i], label, fontsize=10))

# Adjusting text to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

# Setting limits for axes
ax.set_xlim(0, max(x) + 10)
ax.set_ylim(0, max(y) + 10)

# Adding labels and title
ax.set_xlabel("Crash Observed (%)")
ax.set_ylabel("Crash Predicted (%)")
ax.set_title("Plot of the variation of my characteristics given the predicted and observed crash")


# Adding a color bar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Sample Size')

plt.show()


#%% Prepare DS for plots

#--- Age
# unscale age
mean_age = np.mean(car_insu['Age'])
std_age = np.std(car_insu['Age'])

len(car_insu[car_insu['Income'] == 0])
mean_inc = np.mean(car_insu['Income'])
std_inc = np.std(car_insu['Income'])


ds_inc = wds.sort_values(by = 'Income')
ds_inc['Inc_unscaled'] = (ds_inc['Income']*std_inc) + mean_inc
ds_inc['Cum Crash #'] = ((ds_inc['Crash'].cumsum())/len(ds_inc))*100
ds_inc['Crash Cum pred'] =((ds_inc['Predict.'].cumsum())/len(ds_inc))*100

ds_age = wds.sort_values(by = "Age")
ds_age['Age_unscaled'] = (ds_age['Age']*std_age) + mean_age
ds_age['Cum Crash'] = ds_age['Crash'].cumsum()
ds_age['Cum Crash2'] = ((ds_age['Cum Crash'])/len(ds_age))*100
ds_age["Crash Cum Pred"] = ((ds_age['Predict.'].cumsum())/len(ds_age))*100

np.mean(car_insu['Age'])# 44.70
np.max(car_insu['Age'])# 81
np.min(car_insu['Age'])#16


#%% mathplotlib plot


# Assuming ds_age is your DataFrame
plt.figure(figsize=(10, 6))

# Plot the lines
plt.plot(ds_age['Age_unscaled'], ds_age['Cum Crash2'], color='red', label='Observed Crash Proportion')
plt.plot(ds_age['Age_unscaled'], ds_age['Crash Cum Pred'], color='blue', label='Predicted Crash Proportion')

# Adding horizontal lines
plt.axhline(y=26.84, color='red', linestyle='-', linewidth=1)
plt.axhline(y=37.03, color='blue', linestyle='-', linewidth=1)

# Adding vertical line
plt.axvline(x=44.7, color='black', linestyle='dotted', linewidth=1)
plt.text(44.7, 0, 'mean age', va='bottom', ha='center', size=12)

# Annotate text for horizontal lines
plt.text(44.7 - 2.7, 26.84, '26.84', color='red', va='bottom', ha='center', size=12)
plt.text(44.7 - 2.7, 37.03, '37.03', color='blue', va='bottom', ha='center', size=12)

# Set labels and title
plt.xlabel('Age')
plt.ylabel('Proportion of Crash in %')
plt.title('Evolution of the proportion of Crash given the age')

# Customize background
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Light grid lines
plt.gca().set_facecolor('#f0f0f0')  # Light grey background
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.4) 

# show legend
plt.legend()

# Show the plot
plt.show()

#%% Evo prop crash given the income

# Assuming ds_age is your DataFrame
plt.figure(figsize=(10, 6))

# Plot the lines
plt.plot(ds_inc['Inc_unscaled'], ds_inc['Cum Crash #'], color='red', label='Observed Crash Proportion')
plt.plot(ds_inc['Inc_unscaled'], ds_age['Crash Cum Pred'], color='blue', label='Predicted Crash Proportion')

# Adding horizontal lines
plt.axhline(y=26.84, color='red', linestyle='-', linewidth=1)
plt.axhline(y=37.03, color='blue', linestyle='-', linewidth=1)

# Adding vertical line
plt.axvline(x=mean_inc, color='black', linestyle='dotted', linewidth=1)
plt.text(mean_inc, 0, 'mean inc', va='bottom', ha='center', size=12)

# Annotate text for horizontal lines
plt.text(44.7 - 2.7, 26.84, '26.84', color='red', va='bottom', ha='center', size=12)
plt.text(44.7 - 2.7, 37.03, '37.03', color='blue', va='bottom', ha='center', size=12)

# Set labels and title
plt.xlabel('Income')
plt.ylabel('Proportion of Crash in %')
plt.title('Evolution of the proportion of Crash given the income')

# Customize background
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Light grid lines
plt.gca().set_facecolor('#f0f0f0')  # Light grey background
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.4) 

# show legend
plt.legend()

# Show the plot
plt.show()

#%% Additional comparison

hg_ds = car_insu[car_insu['Education'] == 'High School']
hg_ds['Claims Flag (Crash)'].value_counts()
(1346)
#%% question 2
# KerasRegressor has some kind of bugs "Value Error : ... loss"

kr = KerasRegressor(model = l_mod2,epochs = 100, batch_size=512, verbose = 1, #class_weights = class_w2,
                    #callbacks=[tf.keras.callbacks.EarlyStopping('val_loss',#modified to loss
                    #                        patience=5,restore_best_weights = True, mode = "min",
                    #                       verbose = 1)]
                    )
kr.fit(x_train,y_train,validation_data = (x_test,y_test))
# impossible to go further

features = ["Age","Income"]


