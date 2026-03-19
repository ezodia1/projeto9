import pandas as pd

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

print(comparison[['hypothesis', 'rank_ICE', 'rank_RICE', 'variação']].to_string())


#Ao utilizar o método RICE nós adicionamos a variável(fator) alcance ao cálculo, isso nos permite entender quantas pessoas serão afetadas pela hipótese que queremos testar. Ao incluir o alcance nós temos uma melhor perspectiva  de como isso vai afetar nosso público alvo, por esse motivo hipóteses que tinham um alcance menor acabaram abaixando no RICE em relação ao ICE enquanto as que alcançavam mais pessoas acabaram subindo na ordem de prioridade