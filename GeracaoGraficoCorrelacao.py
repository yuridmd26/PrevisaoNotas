import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o conjunto de dados
dados_estudantes = pd.read_csv('student-mat.csv', sep=';')

# Criar novas colunas (features)
dados_estudantes['media_notas'] = (dados_estudantes['G1'] + dados_estudantes['G2']) / 2
dados_estudantes['log_faltas'] = np.log1p(dados_estudantes['absences'])

# Definir as colunas numéricas relevantes
colunas_numericas = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 
    'log_faltas', 'media_notas'
]

# Criar um conjunto de dados com as colunas numéricas e a variável alvo (nota final)
dados_numericos_com_alvo = dados_estudantes[colunas_numericas + ['G3']]

# Calcular a matriz de correlação
matriz_correlacao = dados_numericos_com_alvo.corr()

# Plotar o heatmap de correlação
plt.figure(figsize=(12, 8))
sns.heatmap(matriz_correlacao, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Matriz de Correlação entre Variáveis Numéricas e Nota Final (G3)")
plt.show()

# Obter a correlação de cada variável com a variável alvo (G3)
correlacao_com_alvo = matriz_correlacao["G3"].sort_values(ascending=False)

# Plotar gráfico de barras para mostrar a correlação com G3
plt.figure(figsize=(10, 5))
correlacao_com_alvo.drop('G3').plot(kind='bar', color='skyblue', edgecolor='black')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("Correlação das Variáveis Numéricas com a Nota Final (G3)")
plt.ylabel("Coeficiente de Correlação")
plt.xlabel("Variáveis")
plt.show()
