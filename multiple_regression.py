import statsmodels.api as sm
import streamlit as st

def perform_multiple_regression(df):
    st.subheader("Multiple Linear Regression Analysis")

    # Description of the analysis
    st.markdown("""
    This application performs a multiple linear regression analysis to understand the relationship between different variables and the EPDS score.
    Select the independent variables and the dependent variable to view the results.
    """)

    # Select independent and dependent variables
    all_columns = df.columns.tolist()
    dependent_var = st.selectbox('Select the dependent variable', all_columns, index=all_columns.index('EPDS_SCORE'))
    independent_vars = st.multiselect('Select the independent variables', all_columns, default=['Sleep_hours', 'night_awakening_number_bb1', 'Marital_status_edit'])

    if independent_vars:
        X = df[independent_vars]
        y = df[dependent_var]

        # Add constant to the model
        X = sm.add_constant(X)

        # Fit the multiple linear regression model
        model = sm.OLS(y, X).fit()

        # Display model summary
        st.subheader("Model Summary")
        model_summary = model.summary().as_text()
        st.text_area("Model Summary", model_summary, height=400)

        # Add explanatory dictionary
        st.subheader("Interpretation of Results")
        st.markdown("""
        **Output Dictionary:**

        - **Dep. Variable**: The dependent variable being predicted by the model.
        - **R-squared**: The proportion of variance in the dependent variable explained by the independent variables. Values closer to 1 indicate a better fit.
        - **Adj. R-squared**: The R-squared adjusted for the number of variables in the model. It is a more accurate measure of the model fit.
        - **F-statistic**: F-test for the overall significance of the model. Higher values indicate that at least one independent variable is significant.
        - **Prob (F-statistic)**: The p-value associated with the F-test. Values less than 0.05 indicate that the model is statistically significant.
        - **Log-Likelihood**: The log-likelihood of the model. Higher values indicate a better fit.
        - **AIC/BIC**: Akaike and Bayesian information criteria. Lower values indicate a better model.
        - **Df Residuals**: The number of degrees of freedom of the residuals.
        - **Df Model**: The number of degrees of freedom of the model.
        - **coef**: The estimated coefficients for each independent variable. Indicate the expected change in the dependent variable for a one-unit change in the independent variable.
        - **std err**: The standard error of the coefficients. Measure of the precision of the coefficient estimates.
        - **t**: The t-value for the significance test of the coefficients.
        - **P>|t|**: The p-value associated with the t-test. Values less than 0.05 indicate that the coefficient is statistically significant.
        - **[0.025, 0.975]**: The 95% confidence interval for the coefficients.
        - **Omnibus**: Test for normality of the residuals. Higher values indicate that the residuals are not normally distributed.
        - **Prob(Omnibus)**: The p-value associated with the Omnibus test. Values less than 0.05 indicate that the residuals are not normally distributed.
        - **Durbin-Watson**: Test for autocorrelation of the residuals. Values close to 2 indicate no autocorrelation.
        - **Jarque-Bera (JB)**: Test for normality of the residuals. Higher values indicate that the residuals are not normally distributed.
        - **Prob(JB)**: The p-value associated with the Jarque-Bera test. Values less than 0.05 indicate that the residuals are not normally distributed.
        - **Skew**: The skewness of the residuals. Values different from 0 indicate skewness.
        - **Kurtosis**: The kurtosis of the residuals. Values different from 3 indicate abnormal kurtosis.
        - **Cond. No.**: The condition number of the model. Higher values indicate multicollinearity problems.
        """)

    else:
        st.write("Please select at least one independent variable.")