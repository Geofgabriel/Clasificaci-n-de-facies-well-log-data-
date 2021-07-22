# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import geo_plots as gp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm

filename = 'facies_vectors.csv'
training_data = pd.read_csv(filename)

# Me quedo con uno de los pozos para probar al final:
blind = training_data[training_data['Well Name'] == 'SHANKLE']
training_data = training_data[training_data['Well Name'] != 'SHANKLE']

# 
training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()

#=============================================================0
# Colores para las facies. De esta manera van a ser consistentes a
# lo largo del trabajo.
# FACIES:
# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=bafflestone

facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']

facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
training_data.loc[:,'FaciesLabels'] = training_data.apply(lambda row: label_facies(row, 
                 facies_labels), axis=1)

# Quitamos los valores nulos:
PE_mask = training_data['PE'].notnull().values
training_data = training_data[PE_mask]

# Graficamos los datos que tenemos para uno de los pozos:
l = 0
if l == 1:
    gp.make_facies_log_plot(
            training_data[training_data['Well Name'] == 'SHRIMPLIN'],
            facies_colors)

#CONDICIONAMIENTO DE LOS DATOS: nos quedamos con los features para la clasificacion
correct_facies_labels = training_data['Facies'].values
# quitamos lo que no nos interesa ahora.
feature_vectors = training_data.drop(['Formation', 
                                      'Well Name', 'Depth','Facies','FaciesLabels'], axis=1)

#==============================================================================
# Acondicionamiento de los datos:
#==============================================================================
    
scaler = preprocessing.StandardScaler().fit(feature_vectors)
scaled_features = scaler.transform(feature_vectors)

# Separamos en datos para entrenar y para testear:
X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, correct_facies_labels, test_size=0.2, random_state=42)

#==============================================================================
# CLASIFICADORES:
#==============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier


LAYERS_SIZES = (100, 50) # default es 100
MAX_ITE = 5000
TOLERANCE = 1e-500
ACT_FX= 'logistic'#{identity, logistic, tanh, relu}, # funcion de activación.
n_neighbors = 3 # default es 5

# SUPPORT VECTOR MACHINES
clf = svm.SVC(C = 10, gamma=1)#'auto')
clf.fit(X_train,y_train)
# K-NEARES NEIGHBORS
modelo = KNeighborsClassifier(n_neighbors)
modelo.fit(X_train,y_train)
# LOGISTIC REGRESSION
modelo_lr = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial')
modelo_lr.fit(X_train, y_train)
# RED NEURONAL
modelo_red= MLPClassifier(hidden_layer_sizes=LAYERS_SIZES, max_iter=MAX_ITE,
                      tol= TOLERANCE, verbose= False, activation= ACT_FX)
modelo_red.fit(X_train,y_train)

# Ya entrenados los clasificadores podemos realizar las predicciones de las fases.
predicted_labels_svm = clf.predict(X_test)
predicted_labels_knn = modelo.predict(X_test)
predicted_labels_lr  = modelo_lr.predict(X_test)
predicted_labels_red = modelo_red.predict(X_test)


print('-----------------SVM------------------')
conf_svm = confusion_matrix(y_test, predicted_labels_svm)

p_svm,r_svm,f1_svm = display_cm(conf_svm, facies_labels, display_metrics=True, hide_zeros=True)

metrics_svm = [p_svm,r_svm,f1_svm]
np.savetxt('metricas_svm_c10_gamma1.txt',metrics_svm)

gp.plot_confusion_matrix(y_test, predicted_labels_svm,classes=facies_labels,
                         title='Confusion_matrix_SVM_c10_gamma1')
print('-----------------KNN------------------')
conf_knn = confusion_matrix(y_test, predicted_labels_knn)
p_knn,r_knn,f1_knn = display_cm(conf_knn, facies_labels, display_metrics=True, hide_zeros=True)

metrics_knn = [p_knn,r_knn,f1_knn]
np.savetxt('metricas_knn_8.txt',metrics_knn)

gp.plot_confusion_matrix(y_test, predicted_labels_knn,classes=facies_labels,
                         title='Confusion_matrix_KNN_8')

print('-----------------Logistic regression------------------')

conf_lr = confusion_matrix(y_test, predicted_labels_lr)
p_lr,r_lr,f1_lr = display_cm(conf_lr, facies_labels, display_metrics=True, hide_zeros=True)

metrics_lr = [p_lr,r_lr,f1_lr]
np.savetxt('metricas_lr.txt',metrics_lr)

gp.plot_confusion_matrix(y_test, predicted_labels_lr,classes=facies_labels,
                         title='Confusion_matrix_LR_default')

print('-----------------Neuronal network------------------')
conf_red = confusion_matrix(y_test, predicted_labels_red)
p_red,r_red,f1_red = display_cm(conf_red, facies_labels, display_metrics=True, hide_zeros=True)

metrics_red = [p_red,r_red,f1_red]
np.savetxt('metricas_red_100_50_10.txt',metrics_red)

gp.plot_confusion_matrix(y_test, predicted_labels_red,classes=facies_labels,
                         title='Confusion_matrix_RN_100_50_10')

#==============================================================================
# PROBAMOS CON DATOS "NUEVOS"
#==============================================================================
y_blind = blind['Facies'].values
well_features = blind.drop(['Facies', 'Formation', 'Well Name', 'Depth'], axis=1)
X_blind = scaler.transform(well_features)# necesitamos standarizar los nuevos dato

y_pred_svm = clf.predict(X_blind)
y_pred_knn = modelo.predict(X_blind)
y_pred_lr = modelo_lr.predict(X_blind)
y_pred_red= modelo_red.predict(X_blind)

blind['Prediction_svm'] = y_pred_svm
blind['Prediction_knn'] = y_pred_knn
blind['Prediction_lr'] =  y_pred_lr
blind['Prediction_red'] = y_pred_red
#cv_conf = confusion_matrix(y_blind, y_pred)
t = 0
if t == 1:
    cv_conf_svm = confusion_matrix(y_blind, y_pred_svm)
    cv_conf_knn = confusion_matrix(y_blind, y_pred_knn)
    cv_conf_red = confusion_matrix(y_blind, y_pred_red)

    print('---------------SVM----------------')
    display_cm(cv_conf_svm, facies_labels,
               display_metrics=True, hide_zeros=True)
    print('---------------KNN----------------')
    display_cm(cv_conf_knn, facies_labels,
               display_metrics=True, hide_zeros=True)
    print('---------------RED----------------')
    display_cm(cv_conf_red, facies_labels,
               display_metrics=True, hide_zeros=True)

    gp.compare_facies_plot(blind, 'Prediction_knn', facies_colors)
    gp.compare_facies_plot(blind, 'Prediction_svm', facies_colors)
    gp.compare_facies_plot(blind, 'Prediction_lr', facies_colors)
    gp.compare_facies_plot(blind, 'Prediction_red', facies_colors)

print(' cansado...pero termine de correr :P')