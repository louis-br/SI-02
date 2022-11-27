# Biblioteca para carregar o arquivo csv e manipular seus dados
import pandas as pd

# Bibliotecas para a criação da árvore de decisão
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from utils.utils import print_classifier_metrics
from utils.utils import build_visual_decision_tree
from utils.utils import plot_graph

#===========================================================================================

def main():
    # Treina e testa o modelo da árvore de decisão usando a técnica CART para classificação

    # Carrega o arquivo csv
    sinais_vitais = pd.read_csv('data/sinais_vitais.csv')

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
    print_classifier_metrics(classifier, x_test, y_test, test_results)

    # Cria uma representação gráfica da árvore de decisão encontrada e salva-a como pdf
    build_visual_decision_tree(tree, input_data)

    # Plota o gráfico em 3D dos dados
    plot_graph(input_data, output_classes)

#===========================================================================================

if __name__ == "__main__":
    main()