# Bibliotecas para a criação da árvore de decisão
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from utils.utils import load_data
from utils.utils import print_regressor_metrics
from utils.utils import plot_graph
from utils.utils import plot_results_graph
from utils.utils import save_model

#===========================================================================================

def main():
    # Treina e testa o modelo da árvore de decisão usando a técnica CART para regressão

    # Carrega o arquivo csv em valores de entrada e de saída
    input_data, output_values = load_data('data/sinais_vitais.csv', ['qPA', 'pulso', 'resp'], ['grav'])

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
    plot_results_graph('grav', x_test, y_test, test_results)
    #plot_graph(input_data, output_values)
    
    # Salva o modelo
    save_model(regressor, 'models/cart_regressor')


#===========================================================================================

if __name__ == "__main__":
    main()
