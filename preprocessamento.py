import pandas as pd

def translate_values(df):
    # Convert categorical variables
    marital_status_map = {
        1: 'Single',
        2: 'In a relationship',
        3: 'Separated, divorced or widow.'
    }
    education_map = {
        1: 'No education',
        2: 'compulsory school',
        3: 'Post-compulsory education',
        4: 'UAS or  UT Degree',
        5: 'University'
    }
    how_falling_asleep_map = {
        1: 'While being fed',
        2: 'While being rocked',
        3: 'While being held',
        4: 'Alone in the crib',
        5: 'In the crib with parental presence'
    }

    # Add categorical column for HADS_SCORE
    def categorize_hads(score):
        if score <= 7:
            return 'Unlikely'
        elif 8 <= score <= 11:
            return 'Possible'
        else:
            return 'Probable'

    df['HADS_Category'] = df['HADS_SCORE'].apply(categorize_hads)
    df['Marital_status'] = df['Marital_status'].map(marital_status_map)
    df['Education'] = df['Education'].map(education_map)
    df['sex_baby1'] = df['sex_baby1'].map({1: 'Male', 2: 'Female'})
    df['how_falling_asleep_bb1'] = df['how_falling_asleep_bb1'].map(how_falling_asleep_map)

    return df

def preprocess(df):
    # Remove missing data encoded as "99:99"
    df = df[df['Sleep_night_duration_bb1'] != '99:99']

    # Calculate scores
    df['EPDS_SCORE'] = df[[f'EPDS_{i}' for i in range(1, 11)]].sum(axis=1)
    df['HADS_SCORE'] = df[['HADS_1', 'HADS_3', 'HADS_5', 'HADS_7', 'HADS_9', 'HADS_11', 'HADS_13']].sum(axis=1)

    # Correct CBTS columns (naming changes after 12)
    cbts_columns = [f'CBTS_M_{i}' for i in range(3, 13)] + [f'CBTS_{i}' for i in range(13, 23)]
    df['CBTS_SCORE'] = df[cbts_columns].sum(axis=1)

    # Convert sleep duration
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