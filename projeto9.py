import pandas as pd

#ATRIBUIÇÃO DE DATAFRAMES

df_hypo = pd.read_csv('hypotheses_us.csv')
df_orders = pd.read_csv('orders_us.csv')
df_visits = pd.read_csv('visits_us.csv')

df_visits.info()