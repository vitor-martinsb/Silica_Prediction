# Predição de Qualidade em Processos de Mineração

Este repositório contém dois projetos principais relacionados à análise e predição de qualidade em processos de mineração. Os projetos utilizam dados de um processo de flotação para prever a concentração de sílica em um processo de mineração.

## Estrutura do Repositório

```
.
├── Predicao de Qualidade.ipynb  # Notebook para análise e modelagem de predição de qualidade
├── silica_prediction.py         # Script Python com classes e funções para análise e predição
├── data/                        # Diretório contendo os dados utilizados nos projetos
│   └── MiningProcess_Flotation_Plant_Database.csv  # Base de dados do processo de mineração
```

## Descrição dos Arquivos

### 1. `Predicao de Qualidade.ipynb`
Notebook interativo que realiza as seguintes etapas:
- Carregamento e limpeza dos dados
- Análise exploratória de dados (EDA), incluindo:
  - Estatísticas descritivas
  - Visualização de distribuições
  - Análise de correlação
  - Identificação de outliers
- Modelagem preditiva utilizando:
  - Regressão Linear
  - Regressão Polinomial
  - Regressão com Árvore de Decisão
- Avaliação de desempenho dos modelos com métricas como R² e RMSE

### 2. `silica_prediction.py`
Script Python modular que contém a classe `SilicaPrediction` para realizar as seguintes tarefas:
- Carregamento e pré-processamento dos dados
- Geração de gráficos de análise exploratória, como matriz de correlação
- Treinamento e avaliação de modelos de aprendizado de máquina, incluindo:
  - Regressão Linear
  - Regressão Polinomial
  - Árvore de Decisão
- Diagnóstico de overfitting e underfitting
- Recomendação de modelos não-lineares para melhorar a performance

### 3. `data/MiningProcess_Flotation_Plant_Database.csv`
Arquivo CSV contendo os dados do processo de mineração, incluindo variáveis operacionais e as concentrações de ferro e sílica.

## Requisitos

- Python 3.8 ou superior
- Bibliotecas Python:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn

## Como Usar

### 1. Notebook Interativo
1. Abra o arquivo `Predicao de Qualidade.ipynb` em um ambiente Jupyter Notebook ou no VS Code.
2. Execute as células sequencialmente para realizar a análise exploratória e a modelagem preditiva.

### 2. Script Python
1. Certifique-se de que o arquivo de dados `MiningProcess_Flotation_Plant_Database.csv` está no diretório `data/`.
2. Importe a classe `SilicaPrediction` do arquivo `silica_prediction.py`.
3. Utilize os métodos da classe para carregar os dados, realizar a análise e treinar os modelos. Exemplo:

```python
from silica_prediction import SilicaPrediction

# Inicializar a classe com o caminho do arquivo de dados
silica_pred = SilicaPrediction(data_path='data/MiningProcess_Flotation_Plant_Database.csv')

# Carregar e pré-processar os dados
data = silica_pred.load_data()
X_train, y_train, X_test, y_test = silica_pred.preprocess_data(data)

# Treinar e avaliar o modelo
model, rmse, y_pred = silica_pred.train_model(X_train, y_train, X_test, y_test)
```

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests para melhorias ou correções.

## Licença
Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.