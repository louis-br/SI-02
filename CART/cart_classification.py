# Biblioteca para carregar o arquivo csv e manipular seus dados
import pandas as pd

# Bibliotecas para a criação da árvore de decisão
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Biblioteca para criar um gráfico da árvore de decisão e salvá-lo como pdf
import graphviz

# Biblioteca para plotar um gráfico em 3D dos dados 
import matplotlib.pyplot as plt

#===========================================================================================

def main():
    # Treina e testa o modelo da árvore de decisão usando a técnica CART para classificação

    # Carrega o arquivo csv
    sinais_vitais = pd.read_csv('sinais_vitais.csv')

    # Criar uma matriz apenas com as colunas aPA, pulso, resp e o risco, que são as classes
    sinais_vitais = sinais_vitais.loc[:, ['qPA', 'pulso', 'resp', 'risco']]

    # Cria uma matriz contendo apenas os valores de entrada
    input_data = sinais_vitais.loc[:, ['qPA', 'pulso', 'resp']]

    # Cria uma matriz apenas com os valores de saída, as classes
    output_classes = sinais_vitais.loc[:, ['risco']]

    # Separa os dados de entrada e saída em conjuntos de dados para treinamento e para teste
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_classes, test_size=0.3)

    # Inicializa o classificador CART com o método gini 
    classifier = DecisionTreeClassifier(criterion='gini')

    # Executa o treinamento do modelo, criando a árvore de decisão
    tree = classifier.fit(x_train.values, y_train)

    # Testa o modelo e armazena os resultados do teste
    test_results = classifier.predict(x_test.values)

    # Imprime os resultados de eficácia encontrados
    print_metrics(classifier, x_test, y_test, test_results)

    # Cria uma representação gráfica da árvore de decisão encontrada e salva-a como pdf
    build_visual_decision_tree(tree, input_data)

    # Plota o gráfico em 3D dos dados
    plot_graph(input_data, output_classes)


#===========================================================================================

def print_metrics(classifier, x_test, y_test, test_results):
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
    visual_tree.render('decision_tree_classifier')

#===========================================================================================

def plot_graph(input_data, output_classes):
    # Plota um gráfico em 3D representando as colunas aPA, pulso e resp, 
    # apresenta os dados das classes possíveis como uma variação da cor dos pontos no gráfico 
    graph = plt.figure(figsize=(12, 12))
    ax = graph.add_subplot(projection='3d')
    ax.scatter(input_data.loc[:,'qPA'], input_data.loc[:,'pulso'], input_data.loc[:,'resp'], s=30, c=output_classes)
    plt.show()
    
#===========================================================================================

if __name__ == "__main__":
    main()