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

file_path = "C:/Users/User/Documents/Data Science/LDATS2310 DS for insu and finance/"

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

os.chdir(file_path)
#%% 1 Load data and  Preprocessing

car_insu = pd.read_excel("01Data/Car Insurance Claim.xlsx",index_col="ID")
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
#%% Loading the model and WDS

#Creating unsureness label
nc = x_train.shape[1]
last = l_mod2(nc)
last.load_weights('Mod_weights/mod_last2.weights.h5')
# load the model
wds = pd.read_pickle('wds_sure007.pkl')

#%% Creating the table to analyze
to_inves = ["Gender","Car Use","Urb","Lics R","Sing. Par","Marital Status"]
to_inves = [*to_inves,*list(e_dum.columns),*list(c_dum.columns)]
pre_a = make_table(to_inves,wds)
bbb = get_info((wds["High School"] == 0) & (wds["Masters"] == 0) & (wds["PhD"] == 0),wds)
ccc = get_info((wds["Panel Truck"] == 0) & (wds["Pickup"] == 0) & (wds["SUV"] == 0) & (wds['Sports Car'] == 0) & (wds["Van"] == 0),wds)
analyze = pd.concat((pre_a,bbb,ccc),axis = 0)
analyze.iloc[14,0] = "Bachelors"
analyze.iloc[15,0] = "Minivan"

del(pre_a,bbb,ccc,to_inves)

#%% Question 3 Local Interpretable Model-Agnostic Explanations.

from lime import lime_tabular
# put x_train and test into pd.Dataframe to have column's names.

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
x_train.columns = list(wds.iloc[:,:27].columns)
x_test.columns = list(wds.iloc[:,:27].columns)

cat_feature = [*list(e_dum.columns.values),*list(c_dum.columns.values),"Marital Status","Gender","Sing. Par","Car Use","Urb","Lics R"]
expl = lime_tabular.LimeTabularExplainer(np.asarray(x_train),mode = "regression",
                                         feature_names=x_train.columns.values,
                                         categorical_names = cat_feature, kernel_width=np.sqrt(27)*0.20,
                                         random_state=4243)

#-- Randomly sample 4 missclassified points

# Filter subsets based on the conditions
sub1 = deepcopy(wds[(wds['Crash'] == 0) & (wds['Predict.'] == 1) & (wds['Sure'] == 0)])
sub2 = deepcopy(wds[(wds['Crash'] == 0) & (wds['Predict.'] == 1) & (wds['Sure'].isin([-2, 2]))])
sub3 = deepcopy(wds[(wds['Crash'] == 1) & (wds['Predict.'] == 0) & (wds['Sure'] == 0)])
sub4 = deepcopy(wds[(wds['Crash'] == 1) & (wds['Predict.'] == 0) & (wds['Sure'].isin([-2, 2]))])

# Randomly select one point from each subset
point1 = sub1.sample(n=1).index[0]
point2 = sub2.sample(n=1).index[0]
point3 = sub3.sample(n=1).index[0]
point4 = sub4.sample(n=1).index[0]

print(f'{point1} \t {point2} \t {point3} \t {point4} ')

wds.iloc[625,:]
del()
#%% Subset 1 analysis
from statsmodels.stats.proportion import proportions_ztest

explanation = expl.explain_instance(np.array(x_test.iloc[point1,]),
                                    last.predict, num_features=10, num_samples=1000)

with plt.style.context("ggplot"):
    explanation.as_pyplot_figure()
    
explanation.save_to_file('plots/lime_explanation_p1.html')
explanation.as_list()
explanation.as_map()
print("Explanation Local Prediction  : ", explanation.local_pred)
print("Explanation Global Prediction : ", explanation.predicted_value)
wds.iloc[133,:]
wds['Income'].describe() # mean -0.0131
wds['Vehicle Value'].describe() # mean 0.0145
sub1['Income'].describe()

# inspect sub1
Counter(car_insu['Gender'])['F'] #
np.mean(sub1['Gender']) #0.4183
np.mean(sub1['Marital Status']) #0.5048
np.mean(sub1['Car Use']) # 0.4904
np.mean(sub1['Urb']) #0.9615 !! Interesting
Counter(car_insu['City Population']) # 0.7919 of Urb

np.mean(sub1['Lics R']) # 0.1490
Counter(car_insu['License Revoked']) # .1244 of NO

np.mean(sub1['Sing. Par']) #0.1683

# stat diff in women proportion
p1 = Counter(car_insu['Gender'])['F']/(Counter(car_insu['Gender'])['F'] + Counter(car_insu['Gender'])['M']) 
p2 = 1 - np.mean(sub1['Gender'])

p1 = Counter(car_insu['City Population'])['Urban']/(Counter(car_insu['City Population'])['Urban'] + Counter(car_insu['City Population'])['Rural']) 
p2 = np.mean(sub1['Urb'])

n1 = len(car_insu)
n2 = len(sub1)

successes = [p1 * n1, p2 * n2]

# Total number of observations in each group
nobs = [n1, n2]
z_stat, p_value = proportions_ztest(successes, nobs)

print(f"P-value: {p_value}")

if p_value < 0.05:
    print("The difference in proportions is statistically significant.")

#%% Point 2
import scipy.stats as stats
explanation = expl.explain_instance(np.array(x_test.iloc[point2,]),
                                    last.predict, num_features=10, num_samples=1000)

with plt.style.context("ggplot"):
    explanation.as_pyplot_figure()
    
explanation.show_in_notebook()
explanation.save_to_file('plots/lime_explanation_p2.html')
explanation.as_list()
explanation.as_map()
print("Explanation Local Prediction  : ", explanation.local_pred)
print("Explanation Global Prediction : ", explanation.predicted_value)#%% Point 2

wds.iloc[625,:27] == x_test.iloc[625,:]
wds.iloc[625,]
wds['Income'].describe() # mean -0.0131
sub2['Income'].describe()

wds['Travel Time'].describe()
sub2['Travel Time'].describe()
car_insu['Travel Time'].describe()
x_train['Travel Time'].describe()


# Analyze the whole subset
mean_inc = np.mean(car_insu['Income'])
std_inc = np.std(car_insu['Income'])
pd.set_option('display.float_format', '{:.2f}'.format)

sub2['Inc_unscaled'] = (sub2['Income'] * std_inc) + mean_inc
sub2['Inc_unscaled'].describe()
car_insu['Income'].describe()

# Assuming sub2['Inc_unscaled'] and car_insu['Income'] are your two samples
mean_diff_test = stats.ttest_ind(sub2['Inc_unscaled'], car_insu['Income'], equal_var=False)

# Output the t-test result
print(f'T-statistic: {mean_diff_test.statistic}')
print(f'P-value: {mean_diff_test.pvalue}') # no statistical difference

np.mean(sub2['Gender'])
np.mean(sub2['Urb']) #0.9444

p1 = Counter(car_insu['City Population'])['Urban']/(Counter(car_insu['City Population'])['Urban'] + Counter(car_insu['City Population'])['Rural']) 
p2 = np.mean(sub2['Urb'])

n1 = len(car_insu)
n2 = len(sub2)

successes = [p1 * n1, p2 * n2]

# Total number of observations in each group
nobs = [n1, n2]
z_stat, p_value = proportions_ztest(successes, nobs)

print(f"P-value: {p_value}")
print('Signficantly different')

#%% Point 3
explanation = expl.explain_instance(np.array(x_test.iloc[point3,]),
                                    last.predict, num_features=10, num_samples=1000)

with plt.style.context("ggplot"):
    explanation.as_pyplot_figure()
    
explanation.show_in_notebook()
explanation.save_to_file('plots/lime_explanation_p3.html')
explanation.as_list()
explanation.as_map()
print("Explanation Local Prediction  : ", explanation.local_pred)
print("Explanation Global Prediction : ", explanation.predicted_value)#%% Point 2

wds.iloc[14,:]
wds['Travel Time'].describe()

percentile = stats.percentileofscore(wds['Travel Time'], wds.loc[14,'Travel Time'], kind='rank')
print(f'The value is at the {percentile}th percentile in the distribution of Travel Time.')



#%% Point 4

explanation = expl.explain_instance(np.array(x_test.iloc[point4,]),
                                    last.predict, num_features=10, num_samples=1000)
#points4 == 467
with plt.style.context("ggplot"):
    explanation.as_pyplot_figure()
    
explanation.show_in_notebook()
explanation.save_to_file('lime_explanation_p4.html')
explanation.as_list()
explanation.as_map()
print("Explanation Local Prediction  : ", explanation.local_pred)
print("Explanation Global Prediction : ", explanation.predicted_value)

wds.iloc[467,:]

#%% Shapley's value
import shap
print(np.__version__) # must be <= 1.25.0

# creating shap object with my analyzed rows
x_test = pd.DataFrame(x_test)
x_test.columns = list(wds.iloc[:,:27].columns)

# Specific rows to include
specific_rows = x_test.iloc[[467, 14, 625, 133]]
# Sample the remaining rows excluding the specific ones
remaining_sample = shap.utils.sample(x_test.drop([467, 14, 625, 133]), 96,random_state=4243)
X_100 = pd.concat([specific_rows, remaining_sample])

X_100 = X_100.reset_index(drop=False) # reset index while keeping it
idx = X_100['index'] # keep the corresponding index
X_100.drop('index',axis = 1, inplace = True) # drop it from X_100

# Create the explainer instance
explainer = shap.Explainer(last.predict, X_100,seed = 4243)
shap_val = explainer(X_100)

# first fourth of x_100 are my point
# check idx

#%% analyze 2
shap.initjs()

# as time consuming, we use a sub-sample of 100 data
X_100      = shap.utils.sample(x_train.astype(float), 100)
y_100_pred = last.predict(X_100)
explainer   = shap.Explainer(last.predict, X_100)
shap_values = explainer(X_100)

# we focus on one contract
sample_ind = 0 #range(18,20)
X_100 = pd.DataFrame(X_100)
print(X_100.iloc[sample_ind,:])
X_100.columns = list(wds.iloc[:,:27].columns)

ax= plt.figure(figsize=(7,5))
shap.partial_dependence_plot(
    "Income", last.predict, X_100, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:],ax=ax)

ax= plt.figure(figsize=(7,5))
shap.partial_dependence_plot(
    "High School", last.predict, X_100, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:],ax=ax)


ax= plt.figure(figsize=(7,5))
shap.partial_dependence_plot(
    "Vehicle Value", last.predict, X_100, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:],ax=ax)


ax= plt.figure(figsize=(7,5))
shap.partial_dependence_plot(
    "Travel Time", last.predict, X_100, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:],ax=ax)

ax= plt.figure(figsize=(7,5))
shap.partial_dependence_plot(
    "Car Use", last.predict, X_100, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:],ax=ax)

ax= plt.figure(figsize=(7,5))
shap.partial_dependence_plot(
    "Marital Status", last.predict, X_100, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:],ax=ax)

ax= plt.figure(figsize=(7,5))
shap.partial_dependence_plot(
    "Gender", last.predict, X_100, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:],ax=ax)



#%% point 2 #625 

test = shap_val[0]
print(test.shape)
shap.plots.waterfall(shap_val[0]) # unable to make it work
#%% point 3 # 14
shap.plots.waterfall(shap_val[2])

#%% point 4 #133

shap.plots.waterfall(shap_val[3])