import pickle

import numpy as np
import os

import pandas as pd
from keras.models import model_from_json
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def import_model(main_dir, sub_dir, number):
    models = []
    for i in range(number):
        json_f = open(main_dir + sub_dir + "/model_"+str(i)+".json", 'r')
        loaded_model_json = json_f.read()
        json_f.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights((main_dir + sub_dir + "/model_"+str(i)+".h5"))
        models.append(loaded_model)
    return models


def scoring(models, data):
    probas_ = [model.predict(data) for model in models]
    probas_ = [np.mean(scores) for scores in zip(*probas_)]
    return probas_

#########
def run_prediction(models,encoded_data,label="ligand_model_runbinding"):
    folder = "results/cross_model"  # change as your saved folder
    if not os.path.isdir(folder):
        os.makedirs(folder)

    ########
    encoded_pep = np.array([i[0] for i in encoded_data])
    encoded_hla = np.array([i[1] for i in encoded_data])
    encoded_score = np.array([i[2] for i in encoded_data])

    sample_ls = np.random.choice(len(encoded_pep),size=10000,replace=False)
    encoded_pep = encoded_pep[sample_ls]
    encoded_hla = encoded_hla[sample_ls]
    encoded_score = encoded_score[sample_ls]

    ph_pairs = [encoded_pep,encoded_hla]
    ph_scores = scoring(models, ph_pairs)

    true_labels = [1 if score > (1 - np.log(500) / np.log(50000)) else 0 for score in encoded_score]
    df = pd.DataFrame(data={"TrueY": true_labels, "PredY":ph_scores})
    df.to_csv(folder + '/'+label+'_probas.txt',index=False)

    ############
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 16}
    figsize = (7, 7)

    # ROC_figure
    figure, ax = plt.subplots(figsize=figsize)
    ax.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    fpr, tpr, thresholds = roc_curve(true_labels, ph_scores)
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
    figure.savefig(folder + '/'+label+'_roc.png', dpi=300, bbox_inches='tight')

    # PR_figure
    figure2, ax2 = plt.subplots(figsize=figsize)
    ax2.tick_params(labelsize=18)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    precision, recall, _ = precision_recall_curve(true_labels, ph_scores)
    ax2.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.4f)' % (average_precision_score(true_labels, ph_scores)),
             lw=2, alpha=.8)

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall', font)
    ax2.set_ylabel('Precision', font)
    title2 = 'PR Curve'
    ax2.set_title(title2, font)
    ax2.legend(loc="lower left")
    figure2.savefig(folder + '/'+label+'_pr.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    with open('data/encoded_allele_peptide_binding.pkl', 'rb') as handle:
        encoded_data = pickle.load(handle)

    model_folder = "results"
    model_ligand = import_model(model_folder, '/ligand_model', 5)
    run_prediction(model_ligand,encoded_data,label="ligand_model_runbinding")

    #############
    with open('data/encoded_allele_peptide_ligand50.pkl', 'rb') as handle:
        encoded_data = pickle.load(handle)

    model_folder = "results"
    model_ligand = import_model(model_folder, '/binding_model', 5)
    run_prediction(model_ligand, encoded_data,label="binding_model_runligand")


