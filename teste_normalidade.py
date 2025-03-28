import pandas as pd
from scipy.stats import shapiro, kstest, anderson

def normality_tests(df, cols):
    results = {}
    summary = []
    for col in cols:
        data = df[col].dropna()
        shapiro_test = shapiro(data)
        ks_test = kstest(data, 'norm', args=(data.mean(), data.std()))
        anderson_test = anderson(data, dist='norm')
        results[col] = {
            'Shapiro-Wilk': shapiro_test,
            'Kolmogorov-Smirnov': ks_test,
            'Anderson-Darling': anderson_test
        }
        # Resumo dos resultados
        is_normal = (shapiro_test.pvalue >= 0.05) and (ks_test.pvalue >= 0.05) and (anderson_test.statistic <= anderson_test.critical_values[2])
        summary.append({'Variable': col, 'Normal': 'SIM' if is_normal else 'NÃO'})
    return results, pd.DataFrame(summary)