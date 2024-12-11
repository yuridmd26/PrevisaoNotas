
# README - Previsão de Notas Finais com XGBoost

## **Visão Geral**
Este repositório contém um código para construir e avaliar um modelo de aprendizado de máquina baseado em **XGBoost**. O objetivo é prever a nota final (**G3**) de estudantes com base em diversas variáveis, como desempenho anterior, características pessoais, hábitos de estudo e frequência às aulas.

## **Estrutura do Código**
O código realiza as seguintes etapas principais:

1. **Carregamento e Tratamento dos Dados**:
   - Leitura do dataset `student-mat.csv`.
   - Tratamento de valores ausentes (substituição por zero).
   - Criação de novas variáveis:
     - `media_notas`: Média das notas intermediárias (G1 e G2).
     - `log_faltas`: Logaritmo do número de faltas para reduzir o impacto de outliers.

2. **Divisão dos Dados**:
   - Separar variáveis independentes (features) e dependente (target: G3).
   - Divisão em conjuntos de treino (80%) e teste (20%).

3. **Pipeline de Processamento e Modelo**:
   - O pipeline inclui:
     - Padronização de variáveis numéricas usando `StandardScaler`.
     - Codificação de variáveis categóricas com `OneHotEncoder`.
     - Integra o modelo `XGBRegressor`.

4. **Ajuste de Hiperparâmetros**:
   - Usa o `GridSearchCV` para encontrar os melhores hiperparâmetros.
   - Avalia o modelo com validação cruzada (5-fold).

5. **Avaliação do Modelo**:
   - Métricas utilizadas:
     - **RMSE** (Root Mean Squared Error).
     - **MAE** (Mean Absolute Error).
     - **R²** (Coeficiente de Determinação).

6. **Cenários de Teste**:
   - Quatro cenários de estudantes são criados para validar a generalização do modelo.

## **Como Usar**

### **1. Requisitos**
Certifique-se de que as seguintes dependências estejam instaladas:
- Python 3.8 ou superior
- Bibliotecas:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`

### **2. Executando o Código**
1. Coloque o arquivo `student-mat.csv` no mesmo diretório do script.
2. Execute o script Python:
   ```bash
   python previsao_notas.py
   ```
3. O script exibirá:
   - Os melhores hiperparâmetros encontrados pelo `GridSearchCV`.
   - Métricas de desempenho no conjunto de teste.
   - Previsões para os cenários de teste.

### **3. Modificando Cenários de Teste**
Os cenários estão definidos no formato de lista de dicionários. Para adicionar ou modificar cenários:
- Altere a lista `cenarios_teste` no código.
- Certifique-se de que todas as colunas estejam presentes.

## **Detalhes do Modelo**

### **Por que XGBoost?**
- **Desempenho Elevado**: O XGBoost utiliza Gradient Boosting com otimizações que garantem alta acurácia.
- **Flexibilidade**: Permite ajustar vários hiperparâmetros para melhorar resultados.
- **Eficiência Computacional**: Otimizado para hardware moderno.

### **Hiperparâmetros Otimizados**
O `GridSearchCV` testa combinações de hiperparâmetros, incluindo:
- `n_estimators`: Número de árvores.
- `max_depth`: Profundidade máxima das árvores.
- `learning_rate`: Taxa de aprendizado.
- `gamma`: Penaliza divisões de baixo ganho informativo.
- `subsample` e `colsample_bytree`: Proporção de dados usados em cada iteração.

## **Resultados**
### **Métricas de Desempenho no Conjunto de Teste**:
- **RMSE**: Avalia o erro médio entre previsões e valores reais.
- **MAE**: Mede o erro absoluto médio.
- **R²**: Indica a proporção da variância explicada pelo modelo.

### **Cenários de Teste**
Foram criados quatro cenários representando perfis de estudantes:
1. Estudante com desempenho mediano e faltas moderadas.
2. Estudante com excelente desempenho acadêmico e sem faltas.
3. Estudante com baixo desempenho e muitas faltas.
4. Estudante com desempenho variável e faltas elevadas.

Os resultados das previsões para esses cenários ajudam a validar a capacidade de generalização do modelo.

## **Contribuições**
Para contribuir:
1. Crie um fork do repositório.
2. Implemente as alterações desejadas.
3. Submeta um pull request com a descrição das modificações.

## **Licença**
Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais informações.
