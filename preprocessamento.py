import pandas as pd


def preprocessamento(df):
    # Remover dados faltantes codificados como "99:99"
    df = df[df['Sleep_night_duration_bb1'] != '99:99']

    # Converter variáveis categóricas
    marital_status_map = {
        1: 'Solteira',
        2: 'Casada/União estável',
        3: 'Outro'
    }
    education_map = {
        1: 'Fundamental',
        2: 'Médio',
        3: 'Superior',
        4: 'Pós-graduação',
        5: 'Outro'
    }

    df['Marital_status'] = df['Marital_status'].map(marital_status_map)
    df['Education'] = df['Education'].map(education_map)
    df['sex_baby1'] = df['sex_baby1'].map({1: 'Masculino', 2: 'Feminino'})

    # Calcular scores
    df['EPDS_SCORE'] = df[[f'EPDS_{i}' for i in range(1, 11)]].sum(axis=1)
    df['HADS_SCORE'] = df[['HADS_1', 'HADS_3', 'HADS_5', 'HADS_7', 'HADS_9', 'HADS_11', 'HADS_13']].sum(axis=1)

    # Corrigir colunas CBTS (a nomenclatura muda após o 12)
    cbts_columns = [f'CBTS_M_{i}' for i in range(3, 13)] + [f'CBTS_{i}' for i in range(13, 23)]
    df['CBTS_SCORE'] = df[cbts_columns].sum(axis=1)

    # Converter duração do sono
    def convert_sleep_duration(time_str):
        try:
            if pd.isna(time_str):
                return None
            hours, minutes = map(int, str(time_str).split(':'))
            return hours + minutes / 60
        except:
            return None

    df['Sleep_hours'] = df['Sleep_night_duration_bb1'].apply(convert_sleep_duration)

    return df