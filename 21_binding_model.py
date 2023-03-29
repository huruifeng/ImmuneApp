import os
import pickle

import h5py
import numpy as np

import random

from math import log
from keras.models import Model, load_model
from keras.layers import Input, Dense, Permute, Flatten, Concatenate, Dot, TimeDistributed, Activation
from keras.layers import LSTM, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

from sklearn.metrics import auc, roc_curve, average_precision_score, precision_recall_curve
from sklearn.model_selection import KFold

np.random.seed(1234)

import matplotlib.pyplot as plt

#################################
def creat_binding_affinity_model():
    pep_filters = 128
    hla_filters = 128
    kernel_size = 2

    pep_input = Input(shape=(26, 20),name="pep_input") ## peptide length
    pep_conv = Conv1D(pep_filters, kernel_size, padding='same', activation='relu', strides=1,name="pep_conv")(pep_input)
    pep_lstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat',name="pep_bilstm")(pep_conv)
    flat_pep = Flatten(name="pep_flatten")(pep_lstm)

    hla_input = Input(shape=(34, 20),name="hla_input")  ## HLA allele length
    hla_conv = Conv1D(hla_filters, kernel_size, padding='same', activation='relu', strides=1, name="hla_conv")(hla_input)
    hla_maxpool = MaxPooling1D(name="hla_maxpoll")(hla_conv)
    hla_lstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat',name="hla_bilstm")(hla_maxpool)
    flat_hla = Flatten(name="hla_flatten")(hla_lstm)

    cat_layer = Concatenate()([flat_pep, flat_hla])
    fc1 = Dense(256, activation="relu")(cat_layer)
    fc2 = Dense(64, activation="relu")(fc1)
    fc3 = Dense(16, activation="relu")(fc2)

    # The attention module
    pep_attention_weights = Flatten()(TimeDistributed(Dense(1))(pep_conv))
    pep_attention_weights = Activation('softmax')(pep_attention_weights)
    pep_conv_permute = Permute((2, 1))(pep_conv)
    pep_attention = Dot(-1)([pep_conv_permute, pep_attention_weights])

    hla_attention_weights = Flatten()(TimeDistributed(Dense(1))(hla_conv))
    hla_attention_weights = Activation('softmax')(hla_attention_weights)
    hla_conv_permute = Permute((2, 1))(hla_conv)
    hla_attention = Dot(-1)([hla_conv_permute, hla_attention_weights])


    merge_layer = Concatenate()([hla_attention, pep_attention, fc3])

    out = Dense(1, activation="sigmoid")(merge_layer)
    model = Model(inputs=[pep_input, hla_input], outputs=out)
    return model

def train_cross_validation(dataset):
    folder = "results/binding_model"  # change as your saved folder
    if not os.path.isdir(folder):
        os.makedirs(folder)

    encoded_pep = np.array([i[0] for i in dataset])
    encoded_hla = np.array([i[1] for i in dataset])
    encoded_score = np.array([i[2] for i in dataset])

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12)

    allprobas = np.array([])
    allylable = np.array([])

    for i_splits, (train_index, val_index) in enumerate(kf.split(encoded_pep)):
        train_pep, train_hla, train_target = encoded_pep[train_index],encoded_hla[train_index],encoded_score[train_index]
        val_pep, val_hla, val_target = encoded_pep[val_index],encoded_hla[val_index],encoded_score[val_index]

        es = EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=5)
        mc = ModelCheckpoint(folder + '/model_%s.h5' % str(i_splits), monitor='val_mse', mode='min', verbose=1, save_best_only=True)

        model = creat_binding_affinity_model()
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        # model.summary()
        plot_model(model, to_file=folder+"/model_plot%s.pdf" % str(i_splits), show_shapes=True,show_layer_activations=True,show_layer_names=True)

        model_json = model.to_json()
        with open(folder + "/model_" + str(i_splits) + ".json", "w") as json_file:
            json_file.write(model_json)

        model.fit([train_pep, train_hla],train_target,
                  batch_size=128,
                  epochs=500,
                  shuffle=True,
                  callbacks=[es, mc],
                  validation_data=([val_pep, val_hla], val_target),
                  verbose=1)

        saved_model = load_model(folder + '/model_%s.h5' % str(i_splits))
        probas = saved_model.predict([val_pep, val_hla])
        val_label = [1 if aff > (1 - log(500) / log(50000)) else 0 for aff in val_target]
        allprobas = np.append(allprobas, probas)
        allylable = np.append(allylable, np.array(val_label))
        del model

    with open(folder + '/Evalution_lable_probas.txt', "w+") as f:
        for j in range(len(allprobas)):
            f.write(str(allylable[j]) + '\t' +str(allprobas[j]) + '\n')

    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16}
    figsize = (7, 7)

    # ROC_figure
    figure, ax = plt.subplots(figsize=figsize)
    ax.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    fpr, tpr, thresholds = roc_curve(allylable, allprobas)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    ax.plot(fpr, tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f)' % (roc_auc),
             lw=2, alpha=.8)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', font)
    ax.set_ylabel('True Positive Rate', font)
    title = 'ROC Curve'
    ax.set_title(title, font)
    ax.legend(loc="lower right")
    figure.savefig(folder + '/5_fold_roc.png', dpi=300, bbox_inches='tight')

    # PR_figure
    figure2, ax2 = plt.subplots(figsize=figsize)
    ax2.tick_params(labelsize=18)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    precision, recall, _ = precision_recall_curve(allylable, allprobas)
    ax2.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.4f)' % (average_precision_score(allylable, allprobas)),
             lw=2, alpha=.8)

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall', font)
    ax2.set_ylabel('Precision', font)
    title2 = 'PR Curve'
    ax2.set_title(title2, font)
    ax2.legend(loc="lower left")
    figure2.savefig(folder + '/5_fold_pr.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    with open('data/encoded_allele_peptide_binding.pkl', 'rb') as handle:
        encoded_data = pickle.load(handle)

    train_cross_validation(encoded_data)
