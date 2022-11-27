# Biblioteca para carregar o arquivo csv e manipular seus dados
import pandas as pd

# Bibliotecas para a criação da árvore de decisão
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from utils.utils import print_regressor_metrics
from utils.utils import plot_graph

def main():
    # Treina e testa o modelo da árvore de decisão usando a técnica CART para regressão

    # Carrega o arquivo csv
    sinais_vitais = pd.read_csv('data/sinais_vitais.csv')

    # Criar uma matriz apenas com as colunas aPA, pulso, resp e grav
    sinais_vitais = sinais_vitais.loc[:, ['qPA', 'pulso', 'resp', 'grav']]

    # Cria uma matriz contendo apenas os valores de entrada
    input_data = sinais_vitais.loc[:, ['qPA', 'pulso', 'resp']]

    # Cria uma matriz apenas com os valores de saída
    output_values = sinais_vitais.loc[:, ['grav']]

    # Separa os dados de entrada e saída em conjuntos de dados para treinamento e para teste
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_values, test_size=0.3)

    # Inicializa o regressor CART
    regressor = DecisionTreeRegressor()

    # Executa o treinamento do modelo, criando a árvore de decisão
    tree = regressor.fit(x_train, y_train)

    # Testa o modelo e armazena os resultados do teste
    test_results = regressor.predict(x_test)

    # Imprime as métricas de eficácia do modelo
    print_regressor_metrics(regressor, x_test, y_test, test_results)

    # Plota o gráfico em 3D dos dados
    plot_graph(input_data, output_values)

#===========================================================================================

if __name__ == "__main__":
    main()