# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:51:00 2019

@author: Natalia
"""

import numpy as np
import matplotlib.pyplot as plt

# cargo los datos de interés
knn = np.loadtxt('metricas_knn_d.txt')
lr = np.loadtxt('metricas_lr_d.txt')
red = np.loadtxt('metricas_red_d.txt')
svm = np.loadtxt('metricas_svm_d.txt')

# Calculo las metricas precision, recall y f1 promedio para cada metodo:

knn_mean = np.mean(knn,axis=1)
print('Precision knn',knn_mean[0])
print('Recall knn',knn_mean[1])
print('F1 knn',knn_mean[2])

lr_mean = np.mean(lr,axis=1)
print('Precision lr',lr_mean[0])
print('Recall lr',lr_mean[1])
print('F1 lr',lr_mean[2])

red_mean = np.mean(red,axis=1)
print('Precision red',red_mean[0])
print('Recall red',red_mean[1])
print('F1 red',red_mean[2])

svm_mean = np.mean(svm,axis=1)
print('Precision svm',svm_mean[0])
print('Recall svm',svm_mean[1])
print('F1 svm',svm_mean[2])

# GRAFICOS de las metricas para visualizar sus performance y comportamiento

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']


fig, ax = plt.subplots(3,1,figsize=(8,8))

ax[0].plot(knn[0,:],'r*',label='KNN')
ax[0].plot(lr[0,:],'b*',label='Lr')
ax[0].plot(red[0,:],'g*',label='Red')
ax[0].plot(svm[0,:],'k*',label='SVM')
ax[0].set_ylabel('Precision')
ax[0].set_xticks([])
ax[0].set_title('Metrics')
ax[0].legend(loc='lower right',fontsize= 'x-small')
ax[0].set_ylim(0,1.05)

ax[1].plot(knn[1,:],'r*')
ax[1].plot(lr[1,:],'b*')
ax[1].plot(red[1,:],'g*')
ax[1].plot(svm[1,:],'k*')
ax[1].set_ylabel('Recall')
ax[1].set_xticks([])
ax[1].set_ylim(0,1.05)

#ax.xaxis.set_major_formatter(FuncFormatter(facies_labels))
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax[2].plot(knn[2,:],'r*')
ax[2].plot(lr[2,:],'b*')
ax[2].plot(red[2,:],'g*')
ax[2].plot(svm[2,:],'k*')
ax[2].set_ylabel('F1')
ax[2].tick_params(axis='x', rotation=70)
ax[2].set_xticks(np.arange(len(facies_labels))) 
ax[2].set_xticklabels(['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS'], fontsize=12)
ax[2].set_ylim(0,1.05)

#fig.savefig('metricas_analisis.pdf',format='pdf',dpi=300,bbox_inches='tight')
