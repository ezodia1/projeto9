#Import de Bibliotecas

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#ATRIBUIÇÃO DE DATAFRAMES

df_hypo = pd.read_csv('hypotheses_us.csv', sep=';')
df_orders = pd.read_csv('orders_us.csv', parse_dates=['date'])
df_visits = pd.read_csv('visits_us.csv', parse_dates=['date'])

#PADRONIZAÇÃO

df_hypo.columns = df_hypo.columns.str.lower().str.replace(' ', '_')
df_orders.columns = df_orders.columns.str.lower().str.replace(' ', '_')
df_visits.columns = df_visits.columns.str.lower().str.replace(' ', '_')

#INFO
#df_hypo.info()
#df_orders.info()
#df_visits.info()

#SUMÁRIO
'''
    HYPOTHESES 

 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   hypothesis  9 non-null      object
 1   reach       9 non-null      int64 
 2   impact      9 non-null      int64 
 3   confidence  9 non-null      int64 
 4   effort      9 non-null      int64 

 
    ORDERS

 #   Column         Non-Null Count  Dtype         
---  ------         --------------  -----         
 0   transactionid  1197 non-null   int64         
 1   visitorid      1197 non-null   int64         
 2   date           1197 non-null   datetime64[ns]
 3   revenue        1197 non-null   float64       
 4   group          1197 non-null   object        

 
    VISITS

 #   Column  Non-Null Count  Dtype         
---  ------  --------------  -----         
 0   date    62 non-null     datetime64[ns]
 1   group   62 non-null     object        
 2   visits  62 non-null     int64         

 '''

#Eliminando usuarios que aparecem nos dois grupos

users_wrong = df_orders.groupby('visitorid')['group'].nunique()
users_contaminated = users_wrong[users_wrong > 1].index

df_orders = df_orders[~df_orders['visitorid'].isin(users_contaminated)]

#Eliminando pedidos com revenue < 0

df_orders = df_orders[df_orders['revenue'] > 0]

#print(df_hypo)
#print(df_orders.sample(10))
#print(df_visits.sample(10))

#PRIORIZAÇÃO DE HIPÓTESES

#Aplicar o framework ICE para priorizar hipóteses. Classifique-os em ordem decrescente de prioridade.

df_hypo['ICE'] = (df_hypo['impact'] * df_hypo['confidence']) / df_hypo['effort']

#print('PRIORIDADE DE TESTE DE HIPOTESE UTILIZANDO ICE')
#print(df_hypo[['hypothesis', 'ICE']].sort_values(by='ICE', ascending=False))

#Aplicar o framework RICE para priorizar hipóteses. Classifique-os em ordem decrescente de prioridade.

df_hypo['RICE'] = (df_hypo['reach'] * df_hypo['impact'] * df_hypo['confidence']) / df_hypo['effort']

#print('PRIORIDADE DE TESTE DE HIPOTESE UTILIZANDO RICE')
#print(df_hypo[['hypothesis', 'RICE']].sort_values(by='RICE', ascending= False))

#Mostre como a priorização de hipóteses muda quando você usa RICE em vez de ICE. Dê uma explicação para as alterações.

# Comparação ICE vs RICE
ice_rank = df_hypo[['hypothesis', 'ICE']].sort_values(by='ICE', ascending=False).reset_index(drop=True)
ice_rank.index += 1
ice_rank.columns = ['hypothesis', 'ICE']
ice_rank['rank_ICE'] = ice_rank.index

rice_rank = df_hypo[['hypothesis', 'RICE']].sort_values(by='RICE', ascending=False).reset_index(drop=True)
rice_rank.index += 1
rice_rank['rank_RICE'] = rice_rank.index

comparison = ice_rank.merge(rice_rank[['hypothesis', 'rank_RICE']], on='hypothesis')
comparison['variação'] = comparison['rank_ICE'] - comparison['rank_RICE']

#print(comparison[['hypothesis', 'rank_ICE', 'rank_RICE', 'variação']].to_string())


#Ao utilizar o método RICE nós adicionamos a variável(fator) alcance ao cálculo, isso nos permite entender quantas pessoas serão afetadas pela hipótese que queremos testar. Ao incluir o alcance nós temos uma melhor perspectiva  de como isso vai afetar nosso público alvo, o método RICE de certa forma penaliza hipóteses que tenham um alto impacto, mas baixo alcance e premia hipóteses que afetem uma ampla gama de usuários por esse motivo hipóteses que tinham um alcance menor acabaram abaixando no RICE em relação ao ICE enquanto as que alcançavam mais pessoas acabaram subindo na ordem de prioridade


   #ANÁLISE DE TESTE A/B


#Faça um gráfico da receita acumulada por grupo. Tire conclusões e crie conjecturas.

total_revenue = df_orders.groupby(['date', 'group'])['revenue'].sum().reset_index()

total_revenue['cumulative'] = total_revenue.groupby('group')['revenue'].cumsum()

#print(total_revenue)

plt.figure(figsize=(12, 8))
sns.lineplot(data=total_revenue, x='date', y='cumulative', hue='group')
plt.savefig('total_revenue.png')
plt.close()

#De um modo geral as receitas de ambos os grupos são muito semelhantes, porém o grupo testado (B) teve um aumento significativo no dia 18/08/2019 o que impulsionou significativamente a receita acumulada

#Faça um gráfico do tamanho médio acumulado do pedido por grupo. Tire conclusões e crie conjecturas.

orders_mean = df_orders.groupby(['date', 'group']).agg(revenue=('revenue', 'sum'), orders=('transactionid', 'count')).reset_index()

orders_mean['cum_revenue'] = orders_mean.groupby('group')['revenue'].cumsum()
orders_mean['cum_orders'] = orders_mean.groupby('group')['orders'].cumsum()

orders_mean['cum_avg_order'] = orders_mean['cum_revenue'] / orders_mean['cum_orders']

plt.figure(figsize=(12,8))
sns.lineplot(data=orders_mean, x='date', y='cum_avg_order', hue='group')
plt.savefig('orders_mean.png')
plt.close()

# Novamente houve um pico no dia 18/08/2019 para o grupo B. Diferente do gráfico de receita acumulada, aqui a diferença entre os grupos é mais evidente, com o grupo B abrindo uma vantagem significativa a partir dessa data. Esse aumento abrupto no ticket médio do grupo B indica a possível presença de outliers puxando a média para cima, e não necessariamente que o grupo B está performando melhor de forma orgânica. Para uma análise mais precisa, será necessário investigar os pedidos de valor extremo nos próximos passos.


#Faça um gráfico da diferença relativa no tamanho médio acumulado do pedido para o grupo B em comparação com o grupo A. Faça conclusões e crie conjecturas.

orders_pivot = orders_mean.pivot(index='date', columns='group', values='cum_avg_order')

orders_pivot['relative_diff'] = (orders_pivot['B'] - orders_pivot['A']) / orders_pivot['A']

orders_pivot['relative_diff'] = orders_pivot['relative_diff'] * 100

#print(orders_pivot)

plt.figure(figsize=(12, 8))
sns.lineplot(data=orders_pivot, x=orders_pivot.index, y='relative_diff')

for date, value in orders_pivot['relative_diff'].items():
    plt.annotate(f'{value:.1f}%', 
                xy=(date, value), 
                xytext=(0, 8), 
                textcoords='offset points',
                ha='center',
                fontsize=7)

plt.axhline(y=0, color='red', linestyle='--')
plt.title('Diferença relativa no tamanho médio acumulado do pedido (B vs A)')
plt.xlabel('Data')
plt.ylabel('Diferença relativa (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('orders_pivot.png')
plt.close()

# A partir desse gráfico é possível perceber que na maior parte do tempo o grupo B esteve com uma margem maior que o grupo A. Mesmo com uma queda no início do teste A/B, de forma geral o grupo B se manteve superior ao grupo A. Porém ainda há o fator atenuante do pico do dia 18/08 que pode estar inflando artificialmente essa diferença devido a possíveis outliers, o que deve ser considerado na tomada de decisão final.

#Calcule a taxa de conversão de cada grupo como a proporção de pedidos para o número de visitas para cada dia. Trace as taxas de conversão diárias dos dois grupos e descreva a diferença. Tire conclusões e crie conjecturas.

#print(df_visits)
#print(df_orders)

orders_count_gb = df_orders.groupby(['date', 'group'])['transactionid'].size().reset_index()
orders_count_gb = orders_count_gb.rename(columns={'transactionid': 'orders_per_day'})
#print(orders_count_gb)

visits_group_gb = df_visits.groupby(['date', 'group'])['visits'].sum().reset_index()
#print(visits_group_gb)

df_conversion_rate = pd.merge(visits_group_gb, orders_count_gb, on=['date', 'group'], how='left')

df_conversion_rate['conversion_rate'] = (df_conversion_rate['orders_per_day'] / df_conversion_rate['visits']) * 100

#print(df_conversion_rate)

plt.figure(figsize=(12, 8))
sns.lineplot(data=df_conversion_rate, x='date', y='conversion_rate', hue='group')

# Média de cada grupo
mean_a = df_conversion_rate[df_conversion_rate['group'] == 'A']['conversion_rate'].mean()
mean_b = df_conversion_rate[df_conversion_rate['group'] == 'B']['conversion_rate'].mean()

plt.axhline(y=mean_a, color='blue', linestyle='--', alpha=0.5, label=f'Média A: {mean_a:.2f}%')
plt.axhline(y=mean_b, color='orange', linestyle='--', alpha=0.5, label=f'Média B: {mean_b:.2f}%')

plt.title('Taxa de conversão diária por grupo')
plt.xlabel('Data')
plt.ylabel('Taxa de conversão (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('conversion_rate.png')
plt.close()

# A partir dos dados da taxa de conversão diária e da média ao longo do mês, percebe-se que o grupo B apresenta uma média de conversão maior que o grupo A. Diferente das análises anteriores, a taxa de conversão não leva em conta o valor dos pedidos, apenas se a visita resultou em compra ou não. Isso significa que os outliers de receita do dia 18/08 não distorcem esse resultado, tornando essa uma evidência mais robusta da superioridade do grupo B.


#Faça um gráfico da diferença relativa na conversão cumulativa para o grupo B em comparação com o grupo A. Tire conclusões e crie conjecturas.