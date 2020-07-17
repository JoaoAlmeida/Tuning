from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gc

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#Nesse programa que queria classificar comentarios (reviews) a partir das notas que as pessoas deram a um hotel. Fulano
#fez um comentario ao hotel e também deu nota 5. Entao em "clean" eu guardo as notas e y tem o texto associado a essas notas
# Quero que meu classificador consiga dar uma nota pra um comentario sem nota.

data = pd.read_pickle("dataFinal")

#Setting up the X and Y data, where X is the review text and Y is the rating
#Three different inputs will be used: original review text, cleaned review text, and only adjectives review text

# original = data.reviewstext
#Nesses dados estao o label de cada
clean = pd.read_pickle("clean")
# adj = pd.read_pickle("adj")

y = DataFrame(0,index=list('reviewsrating'),columns=[1])

#Linhas anteriores foram so para preparar os dados. Em y eu salvo os dados que serão aprendidos pelo modelo. No meu caso, sao
#reviews que pessoas fizeram sobre hoteis
y = data[pd.notnull(data['reviewsrating'])]['reviewsrating']

#Creating a vectorizer to split the text into unigrams and bigrams
vect = TfidfVectorizer(ngram_range = (1,2))
# v_ori = vect.fit_transform(original)
v_clean = vect.fit_transform(clean)
# v_adj = vect.fit_transform(adj)

# Split the dataset. Test size: 25%
X_train, X_test, y_train, y_test = train_test_split(
    v_clean, y, test_size=0.25, random_state=10)

# Set the parameters by cross-validation. Aqui voce passa os parametros a serem testados. Nesse caso estou testando apenas o valor de K
tuned_parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}]

#Vou escolher o modelo com maior precisao. Voce pode mudar isso tbm. Como e um vetor, voce pode testar varias caracteristicas ao mesmo tempo
scores = ['precision']

gcv_mode = "eigen"

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=10,
                       scoring='%s_macro' % score)
    gc.collect()
    clf.fit(X_train, y_train)

    #Aqui ele vai mostrar o melhor valor de k encontrado, e mostrar a precision, recall e f-score para cada label. No meu dataset sao 5
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

pred = clf.predict(X_test)

#Por fim eu ploto uma confusion matrix que e bem util pra visualizar o resultado da classificacao
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]),decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

c2 = confusion_matrix(y_test, pred)

class_names = ['1', '2', '3', '4', '5']

plt.figure()
plot_confusion_matrix(c2, classes=class_names, normalize=False, title='Confusion matrix SVM - Clean Review')

plt.savefig('svm kernel.png')
