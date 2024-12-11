import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Carregar dataset
dados_estudantes = pd.read_csv('student-mat.csv', sep=';')

# Tratamento de valores ausentes
dados_estudantes.fillna(0, inplace=True)

# Criar novas variáveis
dados_estudantes['media_notas'] = (dados_estudantes['G1'] + dados_estudantes['G2']) / 2
dados_estudantes['log_faltas'] = np.log1p(dados_estudantes['absences'])

# Dividir variáveis independentes (features) e dependente (target)
variaveis_independentes = dados_estudantes.drop(['G3', 'G1', 'G2'], axis=1)
variavel_dependente = dados_estudantes['G3']

# Dividir em conjuntos de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(variaveis_independentes, variavel_dependente, test_size=0.2, random_state=42)

# Identificar colunas categóricas e numéricas
colunas_categoricas = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
colunas_numericas = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'log_faltas', 'media_notas']

# Pré-processador para transformar dados
pre_processador = ColumnTransformer(transformers=[('numerico', StandardScaler(), colunas_numericas),('categorico', OneHotEncoder(), colunas_categoricas)])

# Modelo XGBoost
modelo_xgboost = XGBRegressor(random_state=42)

# Pipeline para processamento e treinamento
pipeline_modelo = Pipeline(steps=[('pre_processamento', pre_processador),('modelo', modelo_xgboost)])

# Parâmetros para ajuste de hiperparâmetros
parametros_hiperparametros = {
    'modelo__n_estimators': [200, 300, 400],
    'modelo__learning_rate': [0.01, 0.05],
    'modelo__max_depth': [4, 6, 8],
    'modelo__subsample': [0.8, 0.9],
    'modelo__colsample_bytree': [0.7, 1.0],
    'modelo__gamma': [0, 1, 5],
    'modelo__min_child_weight': [1, 3, 5]
}

# Ajustar hiperparâmetros com GridSearchCV
ajustador_modelo = GridSearchCV(pipeline_modelo, parametros_hiperparametros, cv=5, scoring='r2')
ajustador_modelo.fit(X_treino, y_treino)

# Avaliar modelo ajustado
melhor_modelo = ajustador_modelo.best_estimator_
previsoes_teste = melhor_modelo.predict(X_teste)

# Exibir os melhores parâmetros
print("Melhores parâmetros encontrados pelo GridSearch:")
print(ajustador_modelo.best_params_)

# Métricas de avaliação
rmse = np.sqrt(mean_squared_error(y_teste, previsoes_teste))
mae = mean_absolute_error(y_teste, previsoes_teste)
r2 = r2_score(y_teste, previsoes_teste)

print(f"Resultados no conjunto de teste:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Previsões para cenários de teste
print("\nTreinando o pipeline e preparando os cenários de teste...")

cenarios_teste = [
    {
        'school': 'GP', 'sex': 'F', 'age': 17, 'address': 'U', 'famsize': 'GT3',
        'Pstatus': 'T', 'Medu': 3, 'Fedu': 3, 'traveltime': 2, 'studytime': 3,
        'failures': 0, 'famrel': 4, 'freetime': 3, 'goout': 3, 'Dalc': 2,
        'Walc': 2, 'health': 4, 'absences': 3, 'reason': 'home', 'Mjob': 'teacher',
        'Fjob': 'services', 'guardian': 'mother', 'media_notas': 12, 'log_faltas': np.log1p(3)
    },
    {
        'school': 'GP', 'sex': 'M', 'age': 18, 'address': 'U', 'famsize': 'GT3',
        'Pstatus': 'T', 'Medu': 4, 'Fedu': 4, 'traveltime': 1, 'studytime': 4,
        'failures': 0, 'famrel': 5, 'freetime': 5, 'goout': 4, 'Dalc': 1,
        'Walc': 1, 'health': 5, 'absences': 0, 'reason': 'reputation', 'Mjob': 'health',
        'Fjob': 'other', 'guardian': 'father', 'media_notas': 14, 'log_faltas': np.log1p(0)
    },
    {
        'school': 'MS', 'sex': 'F', 'age': 20, 'address': 'R', 'famsize': 'LE3',
        'Pstatus': 'A', 'Medu': 1, 'Fedu': 0, 'traveltime': 4, 'studytime': 1,
        'failures': 3, 'famrel': 1, 'freetime': 1, 'goout': 1, 'Dalc': 5,
        'Walc': 5, 'health': 1, 'absences': 20, 'reason': 'course', 'Mjob': 'at_home',
        'Fjob': 'at_home', 'guardian': 'other', 'media_notas': 5, 'log_faltas': np.log1p(20)
    },
    {
        'school': 'GP', 'sex': 'M', 'age': 19, 'address': 'U', 'famsize': 'GT3',
        'Pstatus': 'T', 'Medu': 2, 'Fedu': 2, 'traveltime': 2, 'studytime': 4,
        'failures': 2, 'famrel': 2, 'freetime': 2, 'goout': 5, 'Dalc': 4,
        'Walc': 4, 'health': 3, 'absences': 15, 'reason': 'other', 'Mjob': 'services',
        'Fjob': 'teacher', 'guardian': 'mother', 'media_notas': 8, 'log_faltas': np.log1p(15)
    }
]

# Converter cenários para DataFrame
dados_cenarios_teste = pd.DataFrame(cenarios_teste)
dados_cenarios_teste = dados_cenarios_teste.reindex(columns=X_treino.columns, fill_value=0)

# Realizar previsões
previsoes_cenarios = melhor_modelo.predict(dados_cenarios_teste)

print("\nPrevisões para os cenários:")
for i, pred in enumerate(previsoes_cenarios):
    print(f"Cenário {i+1}: Nota prevista (G3) = {pred:.2f}")
