import pandas as pd
from joblib import load
from util.util import print_classifier_metrics, print_regressor_metrics, save_metrics

def main():
    # Carrega os dados de teste enviados pelo professor e testa-os em cada modelo salvo

    cart_regression_model_name = 'cart_regressor'
    mlp_regression_model_name = 'mlp_regressor'
    cart_classification_model_name = 'cart_classifier'
    mlp_classification_model_name = 'mlp_classifier'

    # Carrega os dados enviados pelo professor
    # Dados de input qPA, pulso e resp
    test_input = load_test_data('test_data/input_data.csv', 'input')

    # Carrega os dados de output da coluna grav
    test_grav = load_test_data('test_data/output_data.csv', 'grav')

    # Carrega os dados de output da coluna risco
    test_risco = load_test_data('test_data/output_data.csv', 'risco')

    # Carrega cada um dos quatro modelos salvos
    cart_regression_model = load_model('models/' + cart_regression_model_name)
    mlp_regression_model = load_model('models/' + mlp_regression_model_name)
    cart_classification_model = load_model('models/' + cart_classification_model_name)
    mlp_classification_model = load_model('models/' + mlp_classification_model_name)

    # Testa os dados de entrada com cada um dos modelos e imprime as métricas
    # comparando o resultado encontrado com os dados esperados
    print('[[[ CART Regression ]]]')
    cart_regression_results = cart_regression_model.predict(test_input)
    print_regressor_metrics(cart_regression_model, test_input, test_grav, cart_regression_results)

    print('[[[ MLP Regression ]]]')
    mlp_regression_results = mlp_regression_model.predict(test_input)
    print_regressor_metrics(mlp_regression_model, test_input, test_grav, mlp_regression_results)

    print('[[[ CART Classification ]]]')
    cart_classification_results = cart_classification_model.predict(test_input)
    print_classifier_metrics(cart_classification_model, test_input, test_risco, cart_classification_results)

    print('[[[ MLP Classification ]]]')
    mlp_classification_results = mlp_classification_model.predict(test_input)
    print_classifier_metrics(mlp_classification_model, test_input, test_risco, mlp_classification_results)

    # Salva uma matriz de três colunas por N linhas contendo o id, a gravidade e o risco
    # encontrados nos testes. Separa em dois arquivos: um para os resultados
    # do mlp e outro para os resultados do cart
    save_test_results('results/cart_results', cart_regression_results, cart_classification_results)
    save_test_results('results/mlp_results', mlp_regression_results, mlp_classification_results)
    
    # Salva as métricas encontradas para cada modelo
    save_metrics('results/cart_metrics',
                 cart_regression_model_name,
                 cart_classification_model_name,
                 test_input, 
                 test_grav, 
                 test_risco, 
                 cart_regression_model, 
                 cart_classification_model, 
                 cart_regression_results,
                 cart_classification_results)
    
    save_metrics('results/mlp_metrics',
                 mlp_regression_model_name,
                 mlp_classification_model_name,
                 test_input, 
                 test_grav, 
                 test_risco, 
                 mlp_regression_model, 
                 mlp_classification_model, 
                 mlp_regression_results,
                 mlp_classification_results)


def load_test_data(file_name, data_type):
    # Carrega o arquivo csv
    file_name = file_name.replace('.csv', '')
    sinais_vitais = pd.read_csv(file_name + '.csv')
    
    # Retorna matrizes de dados de acordo com o valor do data_type
    # para 'input' retorna os valores de entrada e para os demais
    # retorna ua matriz com apenas uma coluna contendo os resultados esperados
    if data_type == 'input':
        return sinais_vitais.loc[:, ['qPA', 'pulso', 'resp']]
    elif data_type == 'grav':
        return sinais_vitais.loc[:, ['grav']]
    elif data_type == 'risco':
        return sinais_vitais.loc[:, ['risco']]
    
def load_model(file_name):
    # Carrega um modelo previamente salvo
    file_name = file_name.replace('.joblib', '')
    return load(file_name + '.joblib')


def save_test_results(file_name, grav, risco):
    # Salva os resultados encontrados no teste
    size = len(grav)
    file_name = file_name.replace('.csv', '')

    # Para cada linha, escreve o id, o valor da gravidade e do risco encontrados
    with open(file_name + '.csv', 'w') as file:
        file.write('i,grav,risco\n')
        for i in range(size):
            file.write(str(i + 1) + ',' + str(grav[i]) + ',' + str(risco[i]) + '\n')
    

if __name__ == '__main__':
    main()