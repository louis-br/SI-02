# Biblioteca para carregar o arquivo csv e manipular seus dados
import pandas as pd

# Bibliotecas para a criação da árvore de decisão
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Biblioteca para criar um gráfico da árvore de decisão e salvá-lo como pdf
import matplotlib.pyplot as plt

def main():
    # Treina e testa o modelo da árvore de decisão usando a técnica CART para regressão

    # Carrega o arquivo csv
    sinais_vitais = pd.read_csv('sinais_vitais.csv')

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
    print_metrics(regressor, x_test, y_test, test_results)

    # Plota o gráfico em 3D dos dados
    plot_graph(input_data, output_values)

#===========================================================================================

def print_metrics(regressor, x_test, y_test, test_results):
    # Imprime a acurácia
    print('\n============================== Métricas ==============================')
    print('------------------------------ Acurácia ------------------------------\n')
    print(regressor.score(x_test, y_test))
    
    # Imprime o RSME
    print('\n------------------------------ RSME ------------------------------\n')
    print(mean_squared_error(y_test, test_results))
    print('\n')

#===========================================================================================

def plot_graph(input_data, output_values):
    # Plota um gráfico em 3D representando as colunas aPA, pulso e grav, 
    # apresenta os dados dos valores de output como uma variação da cor dos pontos no gráfico 
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(input_data.loc[:,'qPA'], input_data.loc[:,'pulso'], input_data.loc[:,'resp'], s=30, c=output_values)
    plt.show()

#===========================================================================================

if __name__ == "__main__":
    main()