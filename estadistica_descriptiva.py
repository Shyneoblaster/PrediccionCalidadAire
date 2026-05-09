import pandas as pd

def generar_estadistica(csv_path):
    df = pd.read_csv(csv_path)

    # Configurar formato de visualización como en el ejemplo
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Remover columnas no deseadas (ID y fechas) solo para estadística descriptiva
    columnas_a_remover = ['No', 'year', 'month', 'day', 'hour']
    df_desc = df.drop(columns=[col for col in columnas_a_remover if col in df.columns])

    stats = df_desc.describe().T
    stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats.columns = ['Conteo', 'Promedio', 'Desv. Estándar', 'Mínimo', '25%', 'Mediana', '75%', 'Máximo']

    print('--- Estadística Descriptiva ---')
    print(stats.T.to_string())
    
    print('\n--- Correlación con PM2.5 ---')
    corr = df.corr(numeric_only=True)['pm2.5'].sort_values(ascending=False)
    print(corr.to_string(float_format=lambda x: '%.2f' % x) + '\nName: pm2.5, dtype: float64')

if __name__ == "__main__":
    generar_estadistica('PRSA_data_2010.1.1-2014.12.31.csv')
