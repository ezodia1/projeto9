#Import de Bibliotecas

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats as st

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
plt.title('Receita acumulada por grupo')
plt.xlabel('Data')
plt.ylabel('Receita acumulada (R$)')
plt.grid(True)
plt.tight_layout()
plt.savefig('total_revenue.png')
plt.close()

#De um modo geral as receitas de ambos os grupos são muito semelhantes, porém o grupo testado (B) teve um aumento significativo no dia 18/08/2019 o que impulsionou significativamente a receita acumulada

#Faça um gráfico do tamanho médio acumulado do pedido por grupo. Tire conclusões e crie conjecturas.

orders_mean = df_orders.groupby(['date', 'group']).agg(revenue=('revenue', 'sum'), orders=('transactionid', 'count')).reset_index()

orders_mean['cum_revenue'] = orders_mean.groupby('group')['revenue'].cumsum()
orders_mean['cum_orders'] = orders_mean.groupby('group')['orders'].cumsum()

orders_mean['cum_avg_order'] = orders_mean['cum_revenue'] / orders_mean['cum_orders']

plt.figure(figsize=(12, 8))
sns.lineplot(data=orders_mean, x='date', y='cum_avg_order', hue='group')
plt.title('Tamanho médio acumulado do pedido por grupo')
plt.xlabel('Data')
plt.ylabel('Ticket médio acumulado (R$)')
plt.grid(True)
plt.tight_layout()
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


conversion_cum = df_conversion_rate.pivot(index='date', columns='group', values='conversion_rate')


conversion_cum['cum_a'] = conversion_cum['A'].cumsum()
conversion_cum['cum_b'] = conversion_cum['B'].cumsum()

conversion_cum['relative_diff'] = (conversion_cum['cum_b'] - conversion_cum['cum_a']) / conversion_cum['cum_a'] * 100

#print(conversion_cum)

plt.figure(figsize=(12, 8))
sns.lineplot(data=conversion_cum, x=conversion_cum.index, y='relative_diff')

for date, value in conversion_cum['relative_diff'].items():
    plt.annotate(f'{value:.1f}%', 
                xy=(date, value), 
                xytext=(0, 8), 
                textcoords='offset points',
                ha='center',
                fontsize=7)

plt.axhline(y=0, color='red', linestyle='--')
plt.title('Diferença relativa na conversão cumulativa (B vs A)')
plt.xlabel('Data')
plt.ylabel('Diferença relativa (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('conversion_cum.png')
plt.close()

# Ao analisar o gráfico que mostra a diferença relativa na conversão cumulativa entre os grupos, fica claro que o grupo B apresenta uma taxa de conversão consistentemente maior que o grupo A ao longo do tempo. O único período em que o grupo B ficou abaixo foi nos primeiros dias do teste A/B, o que é esperado dado o volume ainda baixo de dados. Essa análise reforça a superioridade do grupo B sem ser influenciada pelos outliers de revenue identificados anteriormente.


#Calcule os percentis 95 e 99 para o número de pedidos por usuário. Defina o ponto em que um ponto de dados se torna uma anomalia.

orders_by_user = df_orders.groupby('visitorid')['transactionid'].count().reset_index()
orders_by_user = orders_by_user.rename(columns={'transactionid': 'orders'})
orders_by_user_p95 = np.percentile(orders_by_user['orders'], 95)
orders_by_user_p99 = np.percentile(orders_by_user['orders'], 99)

#print(f'Percentil 95: {orders_by_user_p95}')
#print(f'Percentil 99: {orders_by_user_p99}')
#print(orders_by_user.sample(50))

#95% dos usuários fizeram somente 1 pedido, enquanto 99% dos usuários fizeram 2 pedidos ou menos, os usuários que realizaram 3 pedidos já entram como anomalia nesse caso.


#Faça um gráfico de dispersão dos preços dos pedidos. Tire conclusões e crie conjecturas.

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_orders, x='date', y='revenue', hue='group')
plt.title('Dispersão dos preços dos pedidos')
plt.xlabel('Data')
plt.ylabel('Receita (R$)')
plt.grid(True)
plt.tight_layout()
plt.savefig('revenue_dispersion.png')
plt.close()

#A partir desse gráfico de dispersão conseguimos ver dois outliers claros que favorecem o grupo B quando se trata da receita, no dia 13 e no dia 18 houveram dois picos que podem estar afetando a análise quando utilizamos a métrica de revenue.


#Calcule os percentis 95 e 99 dos preços dos pedidos. Defina o ponto em que um ponto de dados se torna uma anomalia.

revenue_p95 = np.percentile(df_orders['revenue'], 95)
revenue_p99 = np.percentile(df_orders['revenue'], 99)
#print(f'Percentil 95: {revenue_p95}')
#print(f'Percentil 99: {revenue_p99}')

#99% dos pedidos tem valor de 830 reais ou menos, o que significa que os dias que teve pedidos de 3000 e 20000 reais foram anomalias que não acontecem regularmente e podem enviesar a análise, o ponto de corte deve ser de 830 reais pra cima com base no percentil 99


#Encontre a significância estatística da diferença na conversão entre os grupos usando os dados brutos. Tire conclusões e crie conjecturas.


grupo_a_conversion = df_conversion_rate[df_conversion_rate['group'] == 'A']['conversion_rate']
grupo_b_conversion = df_conversion_rate[df_conversion_rate['group'] == 'B']['conversion_rate']


stat, p_value = st.mannwhitneyu(grupo_a_conversion, grupo_b_conversion, alternative='two-sided')

print(f'p-value: {p_value}')

if p_value < 0.05:
    print('Diferença estatisticamente significativa')
else:
    print('Diferença não é estatisticamente significativa') 


#O valor P-value se mostrou muito próximo do valor alpha, porém ainda assim não foi grande o suficiente para ter alguma significancia a partir dos dados brutos



#Encontre a significância estatística da diferença no tamanho médio do pedido entre os grupos usando os dados brutos. Tire conclusões e crie conjecturas.


grupo_a_sizemean = df_orders[df_orders['group'] == 'A']['revenue']
grupo_b_sizemean = df_orders[df_orders['group'] == 'B']['revenue']

stat, p_value = st.mannwhitneyu(grupo_a_sizemean, grupo_b_sizemean, alternative='two-sided')

print(f'p-value: {p_value}')

if p_value < 0.05:
    print('Diferença estatisticamente significativa')
else:
    print('Diferença não é estatisticamente significativa')

#A diferença não é estatisticamente significativa, há uma diferença abrupta entre o alpha e o pvalor



#Encontre a significância estatística da diferença na conversão entre os grupos usando os dados filtrados. Tire conclusões e crie conjecturas.

users_to_remove = orders_by_user[orders_by_user['orders'] > 2]['visitorid']
df_orders_filtrado = df_orders[~df_orders['visitorid'].isin(users_to_remove)]

orders_count_filtrado = df_orders_filtrado.groupby(['date', 'group'])['transactionid'].size().reset_index()
orders_count_filtrado = orders_count_filtrado.rename(columns={'transactionid': 'orders_per_day'})

df_conversion_filtrado = pd.merge(visits_group_gb, orders_count_filtrado, on=['date', 'group'], how='left')
df_conversion_filtrado['conversion_rate'] = (df_conversion_filtrado['orders_per_day'] / df_conversion_filtrado['visits']) * 100

grupo_a_filtrado = df_conversion_filtrado[df_conversion_filtrado['group'] == 'A']['conversion_rate']
grupo_b_filtrado = df_conversion_filtrado[df_conversion_filtrado['group'] == 'B']['conversion_rate']

stat, p_value = st.mannwhitneyu(grupo_a_filtrado, grupo_b_filtrado, alternative='two-sided')
print(f'p-value: {p_value}')

if p_value < 0.05:
    print('Diferença estatisticamente significativa')
else:
    print('Diferença não é estatisticamente significativa')

## Após a remoção dos usuários anômalos, o p-value caiu para 0.044, cruzando o limiar de significância de 0.05. Isso confirma que a diferença na taxa de conversão entre os grupos A e B é estatisticamente significativa com os dados filtrados, reforçando que o grupo B apresenta uma conversão genuinamente superior ao grupo A.


#Encontre a significância estatística da diferença no tamanho médio do pedido entre os grupos usando os dados filtrados. Tire conclusões e crie conjecturas.



grupo_a_sizemean_filtrado = df_orders[(df_orders['group'] == 'A') & (df_orders['revenue'] <= revenue_p99)]['revenue']
grupo_b_sizemean_filtrado = df_orders[(df_orders['group'] == 'B') & (df_orders['revenue'] <= revenue_p99)]['revenue']

stat, p_value = st.mannwhitneyu(grupo_a_sizemean_filtrado, grupo_b_sizemean_filtrado, alternative='two-sided')

print(f'p-value: {p_value}')

if p_value < 0.05:
    print('Diferença estatisticamente significativa')
else:
    print('Diferença não é estatisticamente significativa')


#Tome uma decisão com base nos resultados do teste. As decisões possíveis são: 1. Pare o teste, considere um dos grupos o líder. 2. Pare o teste, conclua que não há diferença entre os grupos. 3. Continue o teste.

'''
DECISÃO FINAL: 

Com base nos resultados encontrados após diversos testes, utilizando de dados brutos, dados filtrados e diversas métricas como revenue e conversion_rate, os resultados se mostraram a favor do grupo testado B, mesmo desconsiderando os valores de anomalos, nossa hipótese ainda se mostrou favorável principalmente pelo parametro de taxa de conversão que aumentou no grupo testado durante o período de 1 mês, ambos os grupos foram filtrados e corrigidos, com base nisso eu tomo a decisão de parar o teste e considerar a o teste favorável a B


'''