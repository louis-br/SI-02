# Biblioteca para carregar o arquivo csv e manipular seus dados
import pandas as pd

# Bibliotecas para a criação da árvore de decisão
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

# Biblioteca para criar um gráfico da árvore de decisão e salvá-lo como pdf
import graphviz

# Biblioteca para plotar um gráfico em 3D dos dados 
import matplotlib.pyplot as plt

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
    print('\n============================== Métricas ==============================')
    print('------------------------------ Acurácia ------------------------------\n')
    print(classifier.score(x_test.values, y_test))

    # Imprime as métricas de eficácia do algoritmo
    # Precision, recall, f-measure, acuracidade
    print('\n--------------------------- Outras Métricas --------------------------\n')
    print(classification_report(y_test, test_results))

    # Imprime a matriz de confusão 
    print('------------------------- Matriz de Confusão -------------------------\n')
    print(confusion_matrix(y_test, test_results))
    print('\n')

#===========================================================================================

def print_regressor_metrics(regressor, x_test, y_test, test_results):
    # Imprime a acurácia
    print('\n============================== Métricas ==============================')
    print('------------------------------ Acurácia ------------------------------\n')
    print(regressor.score(x_test, y_test))
    
    # Imprime o RSME
    print('\n------------------------------ RSME ------------------------------\n')
    print(mean_squared_error(y_test, test_results))
    print('\n')

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