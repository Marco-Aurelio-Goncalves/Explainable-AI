# Decifrando a Caixa Preta: Interpretando Modelos de Crédito com LIME

## 1. Contextualização do Problema
No setor bancário, modelos de machine learning são amplamente utilizados para prever o risco de crédito de clientes. Apesar de muitos alcançarem altos índices de acurácia, ainda há um grande desafio: explicar por que o modelo tomou determinada decisão.

Essa falta de transparência gera desconfiança em clientes, dificuldades para gerentes tomarem decisões embasadas e complicações regulatórias, especialmente em casos de negação de crédito.

Este projeto tem como objetivo aplicar técnicas de Explainable AI (XAI), com foco no LIME (Local Interpretable Model-Agnostic Explanations), para tornar as previsões do modelo compreensíveis para diferentes públicos.

## 2. Objetivos
Desenvolver um modelo preditivo de risco de crédito usando Random Forest.

Utilizar o dataset Statlog (German Credit Data) para classificação binária de clientes: bom pagador (1) ou mau pagador (0).

Aplicar o LIME para gerar explicações locais, identificando as variáveis que mais influenciaram cada decisão.

Discutir a importância da interpretabilidade no setor financeiro e limitações da abordagem.

## 3. Dataset Utilizado
Fonte: UCI Machine Learning Repository

Nome: Statlog (German Credit Data)

Tamanho: 1000 instâncias, 20 variáveis preditoras + 1 variável alvo

Atributos de exemplo:

Status da conta

Duração do crédito

Histórico de crédito

Finalidade do crédito

Valor do crédito

Idade

Emprego

Número de créditos existentes

Alvo (Target):

1 → Bom pagador

0 → Mau pagador

## 4. Modelo Preditivo

O modelo escolhido foi o Random Forest Classifier, devido a:

Alta capacidade preditiva para dados tabulares;

Robustez contra overfitting;

Suporte a variáveis categóricas e numéricas.

Etapas de Treinamento:

Pré-processamento:

Codificação de variáveis categóricas com LabelEncoder.

Divisão treino/teste: train_test_split com 80% treino e 20% teste.

Treinamento:

python
Copiar
Editar
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

## 5. Aplicação do LIME
O LIME gera explicações locais, ou seja, interpreta a decisão para um cliente específico.
Funcionamento resumido:

Seleciona-se uma instância (cliente) do conjunto de teste.

O LIME cria pequenas perturbações nos atributos dessa instância.

Ele observa como o modelo reage a essas mudanças.

Ajusta um modelo interpretável simples (ex.: regressão linear) para estimar a decisão localmente.

Retorna um ranking das variáveis mais influentes naquela decisão.

Exemplo no código:

python
Copiar
Editar
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Ruim', 'Bom'],
    mode='classification'
)

exp = explainer.explain_instance(
    X_test.iloc[5],
    model.predict_proba,
    num_features=10
)

## 6. Resultados Obtidos
A explicação para a instância analisada mostrou, por exemplo:

Atributos com maior peso positivo para a aprovação do crédito (como bom histórico de crédito e alto valor de poupança).

Atributos com maior peso negativo que levaram à recusa (como baixa duração no emprego ou histórico ruim).

A visualização gerada pelo LIME facilita a comunicação com:

Clientes → justificando a decisão de forma clara;

Gerentes → oferecendo insights sobre pontos críticos;

Órgãos regulatórios → garantindo conformidade e transparência.



## 7. Limitações e Discussão
Apesar da utilidade, o LIME apresenta algumas limitações:

Sensibilidade ao parâmetro num_features — explicações podem mudar se o número de variáveis exibidas for alterado.

Instabilidade — pequenas variações na instância podem gerar explicações diferentes.

Modelo local ≠ modelo global — o LIME explica apenas um caso específico, não o comportamento global do modelo.

Interpretação depende de contexto — a importância de uma variável deve ser analisada considerando aspectos do domínio de crédito.

## 8. Importância da Interpretabilidade
Confiabilidade: usuários passam a entender como as decisões são tomadas.

Compliance: atende exigências regulatórias, como a Lei Geral de Proteção de Dados (LGPD) e diretrizes do Banco Central.

Melhoria do Modelo: insights podem revelar vieses ou problemas no treinamento.

## 9. Como Reproduzir

### 1. Instalar dependências
pip install pandas numpy scikit-learn lime matplotlib

### 2. Baixar o dataset
Baixe german.data da UCI Repository e salve no diretório do script.

### 3. Executar o script
python lime_credit_model.py

### 4. Ver explicações

No Jupyter/Colab → exp.show_in_notebook()
No arquivo de imagem gerado: explicacao_lime_instancia_5.png

## 10. Referências
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier

Documentação Oficial do LIME

UCI Machine Learning Repository – German Credit Data

