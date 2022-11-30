# Biblioteca para carregar o arquivo csv e manipular seus dados
import pandas as pd
import numpy as np

# Bibliotecas para a criação da árvore de decisão
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_confusion_matrix

# Biblioteca para criar um gráfico da árvore de decisão e salvá-lo como pdf
import graphviz

# Biblioteca para plotar um gráfico em 3D dos dados 
import matplotlib.pyplot as plt

# Biblioteca para salvar um modelo treinado
from joblib import dump, load

#===========================================================================================

def load_data(input_file, input_columns, output_columns):
    # Carrega o arquivo csv
    sinais_vitais = pd.read_csv(input_file)

    # Cria uma matriz contendo apenas os valores de entrada
    input_data = sinais_vitais.loc[:, input_columns]

    # Cria uma matriz apenas com os valores de saída
    output_data = sinais_vitais.loc[:, output_columns]

    # Retorna os dados de entrada e saída
    return input_data, output_data

#===========================================================================================

def print_classifier_metrics(classifier, x_test, y_test, test_results):
    # Imprime apenas a acurácia
    print('\n[[ Métricas ]]\n')
    print('[ Acurácia ]\n')
    print(classifier.score(x_test.values, y_test))

    # Imprime as métricas de eficácia do algoritmo
    # Precision, recall, f-measure, acuracidade
    print('\n[ Precision | Recall | F-Score ]\n')
    print(classification_report(y_test, test_results))

    # Imprime a matriz de confusão 
    print('\n[ Matriz de Confusão ]\n')
    print(confusion_matrix(y_test, test_results))
    #plot_confusion_matrix(classifier, x_test, y_test)
    #plt.show()
    print('\n-------------------------------------------------------------------\n')

#===========================================================================================

def print_regressor_metrics(regressor, x_test, y_test, test_results):
    # Imprime a acurácia
    print('\n[[ Métricas ]]\n')
    print('[ Acurácia ]\n')
    print(regressor.score(x_test, y_test))
    
    # Imprime o RSME
    print('\n[ RSME ]\n')
    print(mean_squared_error(y_test, test_results))
    print('\n-------------------------------------------------------------------\n')

#===========================================================================================

def build_visual_decision_tree(tree, input_data):
    # Desenha a árvore de decisão da árvore encontrada
    tree_data = export_graphviz(tree, out_file=None, 
                                feature_names=input_data.columns, 
                                class_names=['1', '2', '3', '4'],
                                filled=True, 
                                rounded=True, 
                                rotate=True) 
    visual_tree = graphviz.Source(tree_data)

    # Salva o gráfico como um arquivo pdf
    visual_tree.render('CART/decision_tree_classifier')

#===========================================================================================

def plot_graph(input_data, output_classes):
    # Plota um gráfico em 3D representando as colunas aPA, pulso e resp, 
    # apresenta os dados das classes possíveis como uma variação da cor dos pontos no gráfico 
    graph = plt.figure(figsize=(12, 12))
    ax = graph.add_subplot(projection='3d')
    ax.scatter(input_data.loc[:,'qPA'], input_data.loc[:,'pulso'], input_data.loc[:,'resp'], s=30, c=output_classes)
    plt.show()
    
#===========================================================================================

def plot_results_graph(out_column, x_test, y_test, results, tolerance=10):
    # Plota um gráfico em 3D representando as colunas aPA, pulso e resp, 
    # apresenta os dados das classes possíveis como uma variação da cor dos pontos no gráfico,
    # as predições erradas (fora da tolerância) aparecem como um 'x'. 

    results = np.isclose(y_test[out_column], results, atol=tolerance)
    x_correct = x_test[results]
    y_correct = y_test[results]
    results = ~results
    x_incorrect = x_test[results]
    y_incorrect = y_test[results]
    
    graph = plt.figure(figsize=(12, 12))
    ax = graph.add_subplot(projection='3d')
    ax.scatter(x_correct['qPA'], x_correct['pulso'], x_correct['resp'], s=30, c=y_correct, marker='o')
    ax.scatter(x_incorrect['qPA'], x_incorrect['pulso'], x_incorrect['resp'], s=30, c=y_incorrect, marker='x')
    plt.show()
    
#===========================================================================================

def load_model(file_name):
    file_name = file_name.replace('.joblib', '')
    return load(file_name + '.joblib')

#===========================================================================================

def save_model(model, file_name):
    file_name = file_name.replace('.joblib', '')
    dump(model, file_name + '.joblib')

#===========================================================================================

def load_test_data(file_name):
    # Carrega o arquivo csv
    sinais_vitais = pd.read_csv(file_name)

    # Criar uma matriz apenas com as colunas aPA, pulso, resp e o risco, que são as classes
    return sinais_vitais.loc[:, ['qPA', 'pulso', 'resp']]

#===========================================================================================

def save_test_results(file_name, riscos, classes, score, rsme):
    size = 0
    if type(riscos) is np.ndarray:
        size = len(riscos)
    else:
        size = len(classes)

    with open(file_name + '.csv', 'w') as file:
        for i in range(size):
            file.write(str(i + 1) + ',')
            if type(riscos) is np.ndarray:
                file.write(str(riscos[i]) + ',')
            if type(classes) is np.ndarray:
                file.write(str(classes[i]))
            file.write('\n')
    
    with open(file_name + '_results.txt', 'w') as file:
        file.write('score: ' + str(score))

        if rsme != None:
            file.write('\nrsme: ' + str(rsme))

#===========================================================================================

def save_metrics(file_name, regressor_name, classifier_name, test_input, test_grav, test_risco, regressor, classifier, regression_results, classification_results):
    file_name = file_name.replace('.txt', '')

    with open(file_name + '.txt', 'w') as file:
        file.write('Modelo Regressao: ' + regressor_name)
        file.write('\n\nAcuracia da Regressao: \n')
        file.write(str(regressor.score(test_input, test_grav)))
        file.write('\n\nRSME da Regressao: \n')
        file.write(str(mean_squared_error(test_grav, regression_results)))
        file.write('\n\n----------------------------------------------------')
        file.write('\n\nModelo Classificacao: ' + classifier_name)
        file.write('\n\nAcuracia da Classificacao: \n')
        file.write(str(classifier.score(test_input, test_risco)))
        file.write('\n\nPrecision, Recall e F-Score da Classificao: \n')
        file.write(str(classification_report(test_risco, classification_results)))
        file.write('\n\nMatriz de Confusao da Classificacao: \n')
        file.write(str(confusion_matrix(test_risco, classification_results)))
