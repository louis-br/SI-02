# Bibliotecas para a criação da rede neural MLP
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from utils.utils import load_data
from utils.utils import print_classifier_metrics
from utils.utils import plot_results_graph
from utils.utils import save_model

#===========================================================================================

def main():
    for i in range(5):
        run_mlp_classification(i + 1)
    

def run_mlp_classification(i):

    folder = '1'
    # Treina e testa o modelo da rede neural MLP para classificação

    # Inicializa as variáveis que serão utilizadas no modelo classificador:
    #=============================================================================================

    # Número de perceptrons por camada
    hidden_layer_sizes = (200, 100, 50)

    # Número máximo de iterações 
    max_iter = 2000

    # Valor de tolerância do erro. Quando o score não é melhorado em n_iter_no_change iterações, o treinamento é finalizado
    tol = 0.00001

    # Taxa de aprendizagem
    learning_rate_init = 0.0001

    # O tipo de algoritmo utilizado para a otimização da descida de gradiente
    solver = "adam"

    # Função de ativação
    activation = "tanh"

    # Forma com que o learning_rate_init será regulado, ou não, a cada iteração
    learning_rate = "constant"

    # Quantidade de iterações sem alteração no score que indica quando i treinamento deve ser finalizado
    n_iter_no_change = 15

    # Valor que indicará o quanto de informação será impresso na tela enquanto o modelo está sendo treinado
    verbose = 1

    #=============================================================================================

    # Carrega o arquivo csv em valores de entrada e de saída
    input_data, output_classes = load_data('data/sinais_vitais.csv', ['qPA', 'pulso', 'resp'], ['risco'])

    # Separa os dados de entrada e saída em conjuntos de dados para treinamento e para teste
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_classes, test_size=0.3)

    # Inicializa o modelo classificador com as variáveis inicializadas anteriormente
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                               max_iter=max_iter,
                               tol=tol,
                               learning_rate_init=learning_rate_init,
                               solver=solver,
                               activation=activation,
                               learning_rate=learning_rate,
                               verbose=verbose,
                               n_iter_no_change=n_iter_no_change)
    
    # Treina o modelo classificador 
    classifier.fit(x_train, y_train.values.ravel())

    # Testa o modelo treinado com dados de teste
    test_results = classifier.predict(x_test)

    # Imprime a acurácia, a matriz de confusão e outras métricas
    print_classifier_metrics(classifier, x_test, y_test, test_results)

    # Plota o gráfico em 3D dos dados
    #plot_results_graph('risco', x_test, y_test, test_results, tolerance=0)
    
    # Salva o modelo
    save_model(classifier, f'models/mlp/classification/{folder}/{i}', x_test, y_test, test_results, 'classification')

#===========================================================================================

if __name__ == "__main__":
    main()
