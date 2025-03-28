import pandas as pd
import streamlit as st
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

from preprocessamento import preprocess, translate_values
from correlation_kendall_heatmap import calculate_kendall_correlations, plot_heatmap as plot_kendall_heatmap
from correlation_spearman_heatmap import calculate_correlations, plot_heatmap as plot_spearman_heatmap
from multiple_regression import perform_multiple_regression
from teste_normalidade import normality_tests

# Streamlit Interface
st.set_page_config(page_title="Maternal and Infant Health Dashboard", layout="wide")

# Load data
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Preprocessing
df = preprocess(df)
df_translate = translate_values(df.copy())

# Sidebar
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Mother's Age",
                              min_value=int(df_translate['Age'].min()),
                              max_value=int(df_translate['Age'].max()),
                              value=(int(df_translate['Age'].min()), int(df_translate['Age'].max())),
                              step=1)

selected_education = st.sidebar.multiselect("Education",
                                            options=df_translate['Education'].unique(),
                                            default=df_translate['Education'].unique())

# Apply filters
filtered_df_translate = df_translate[
    (df_translate['Age'].between(age_range[0], age_range[1])) &
    (df_translate['Education'].isin(selected_education))
]

# Main visualizations
st.title("Maternal Mental Health and Infant Sleep Analysis")

# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("Average EPDS", f"{filtered_df_translate['EPDS_SCORE'].mean():.1f}")
# with col2:
#     st.metric("Average HADS", f"{filtered_df_translate['HADS_SCORE'].mean():.1f}")
# with col3:
#     st.metric("Average Sleep Hours", f"{filtered_df_translate['Sleep_hours'].mean():.1f}")

# Charts
st.subheader("")
tab1, tab2, tab3, tab4 = st.tabs(["Descriptive Analysis", "Factors Associated with Postpartum Depression (EPDS_SCORE)", "Analyses and Tool", "Normality Tests"])

with tab1:
    st.header("Distribution of Maternal Age")

    # Pie chart with age ranges, percentages, and custom legend
    bins_age = [19, 26, 31, 36, 48]
    labels_age = ['19-25', '26-30', '31-35', '36-47']
    df_translate['Age_Category'] = pd.cut(df_translate['Age'], bins=bins_age, labels=labels_age, right=False)
    age_counts = df_translate['Age_Category'].value_counts()
    fig1, ax1 = plt.subplots()

    # Custom color palette
    chart_colors = ['#2d2e3f', '#ce3450', '#e2d6ca', '#99b8c1', '#e6ba3d', '#91c7bc']

    # Create pie chart with percentages on slices
    patches, texts, autotexts = ax1.pie(age_counts, autopct='%1.1f%%', startangle=140, colors=chart_colors)
    ax1.axis('equal')

    # Create custom legend
    plt.legend(patches, age_counts.index, loc="best")

    st.pyplot(fig1)

    # Section: Distribution of Education Level (Title with the same size)
    st.header("Distribution of Education Level")

    # Bar chart with custom colors and labels
    education_counts = df_translate['Education'].value_counts().sort_index()

    # Increase figure size
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Create bars with custom colors and individual labels
    bars = ax2.bar(education_counts.index, education_counts.values, color=chart_colors)

    ax2.set_xticks(education_counts.index)
    ax2.set_xticklabels(education_counts.index, rotation=0, ha='center')

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    ax2.set_xlabel('Education')
    ax2.set_ylabel('Frequency')

    # Create legend inside the chart
    ax2.legend(loc='best')

    plt.tight_layout()
    st.pyplot(fig2)
########################################################################################################################
    # Section: Marital status
    st.header("Distribution of Marital Status")

    # Count the frequency of each marital status
    marital_status_counts = df['Marital_status_edit'].value_counts()

    # Dictionary mapping numeric values to descriptive labels
    marital_status_labels = {
        1: '1 = Single',
        2: '2 = In a relationship',
        3: '3 = Separated, divorced, or widowed',
        6: '6 = Other'
    }

    # Custom colors for the bars
    bar_colors = ['#ce3450', '#e2d6ca', '#99b8c1', '#e6ba3d']  # Adjust colors as needed

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(marital_status_counts.index, marital_status_counts.values, color=bar_colors)

    # Add legend labels for each bar
    for bar, label in zip(bars, marital_status_counts.index):
        bar.set_label(marital_status_labels[label])

    # Add labels and title
    ax.set_xlabel('Marital Status')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=0, ha='right')

    # Add frequency numbers above each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom')

    # Create the legend inside the chart
    ax.legend(loc='upper right')  # Adjust legend location as needed

    plt.tight_layout()

    # Integrate with Streamlit
    st.pyplot(fig)

    # Baby Age Distribution
    st.header("Baby Age Distribution")

    # Map values to desired labels
    age_categories = {
        1: '≥3 months to <6 months',
        2: '≥6 months to <9 months',
        3: '≥9 months to <12 months'
    }

    # Count the frequency of categories
    category_counts = df_translate['Age_bb'].value_counts().sort_index()

    # Map indices to labels
    category_counts.index = category_counts.index.map(age_categories)

    # Create the bar chart
    fig_age, ax_age = plt.subplots()
    bars = ax_age.bar(category_counts.index, category_counts.values, color=chart_colors)

    # Add frequency above each bar
    for bar in bars:
        yval = bar.get_height()
        ax_age.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    # Configure axis labels
    ax_age.set_xlabel("Baby Age Intervals")
    ax_age.set_ylabel("Frequency")

    # Configure x-axis label rotation
    plt.xticks(rotation=45, ha='right')

    # Create legend in the lower right
    ax_age.legend(bars, category_counts.index, title="Baby Age Intervals", loc='lower right')

    # Display the chart in Streamlit
    st.pyplot(fig_age)

    # Number of Night Awakenings
    st.header("Night Awakenings Frequency")

    # Count the frequency of night awakenings
    awake_counts = df_translate['night_awakening_number_bb1'].value_counts().sort_index()

    # Create the bar chart
    fig_awake, ax_awake = plt.subplots()
    bars = ax_awake.bar(awake_counts.index, awake_counts.values, color=chart_colors)

    # Add frequency above each bar
    for bar in bars:
        yval = bar.get_height()
        ax_awake.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    # Configure axis labels
    ax_awake.set_xlabel("Number of Night Awakenings")
    ax_awake.set_ylabel("Frequency")

    # Remove the legend
    # ax_awake.legend(["Night Awakenings"], loc="best") # Removed

    # Display the chart in Streamlit
    st.pyplot(fig_awake)

    # Infant Sleep Quality
    st.header("Infant Sleep Quality Categories (Fixed Frequencies)")

    # Correct frequencies
    category_frequencies = {
        'Fed': 90,
        'Rocked': 74,
        'Held': 22,
        'Alone': 177,
        'Parental': 74
    }

    # Create DataFrame with frequencies and sort from highest to lowest
    df_frequencies = pd.Series(category_frequencies).sort_values(ascending=False)

    # Define colors for each bar
    bar_colors = ['#2d2e3f', '#ce3450', '#e2d6ca', '#99b8c1',
                  '#e6ba3d']  # Colors for Fed, Alone, Rocked, Parental, Held

    # Create the bar chart
    fig_sleep, ax_sleep = plt.subplots()
    bars = ax_sleep.bar(df_frequencies.index, df_frequencies.values, color=bar_colors)

    # Add frequency above each bar
    for bar in bars:
        yval = bar.get_height()
        ax_sleep.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    # Configure axis labels
    ax_sleep.set_xlabel("Sleep Categories")
    ax_sleep.set_ylabel("Frequency")

    # Create legend with adjusted labels
    legend_labels = {
        'Alone': 'Alone in the crib',
        'Fed': 'While being fed',
        'Rocked': 'While being rocked',
        'Parental': 'In the crib with parental presence',
        'Held': 'While being held'
    }

    ax_sleep.legend(bars, [legend_labels[label] for label in df_frequencies.index], title="Sleep Categories")

    # Display the chart in Streamlit
    st.pyplot(fig_sleep)
########################################################################################################################
    # Additional visualization for HADS_Category
    st.title("Distribution of HADS Categories")

    # Bar chart for HADS categories
    hads_category_counts = df_translate['HADS_Category'].value_counts().reset_index()
    hads_category_counts.columns = ['HADS_Category', 'count']

    fig = px.bar(hads_category_counts,
                 x='HADS_Category', y='count',
                 color='HADS_Category',
                 color_discrete_sequence=chart_colors,
                 labels={'HADS_Category': 'Category', 'count': 'Count'},
                 title='Distribution of HADS Categories')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.title("Factors Associated with Postpartum Depression (EPDS_SCORE)")

    # Interactive box plots with trend line
    st.subheader("Relationship between EPDS_SCORE and Independent Variables")

    # Sleep_hours vs EPDS_SCORE
    fig1 = px.box(filtered_df_translate, x='Sleep_hours', y='EPDS_SCORE',
                  title='EPDS_SCORE vs Sleep_hours',
                  labels={'Sleep_hours': 'Sleep Hours', 'EPDS_SCORE': 'EPDS_SCORE'})
    x_vals = filtered_df_translate['Sleep_hours']
    y_vals = filtered_df_translate['EPDS_SCORE']
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    fig1.add_traces(px.line(x=x_vals, y=p(x_vals), labels={'x': 'Sleep Hours', 'y': 'EPDS_SCORE'}).data)
    st.plotly_chart(fig1, use_container_width=True)

    # night_awakening_number_bb1 vs EPDS_SCORE
    fig2 = px.box(filtered_df_translate, x='night_awakening_number_bb1', y='EPDS_SCORE',
                  title='EPDS_SCORE vs Number of Night Awakenings',
                  labels={'night_awakening_number_bb1': 'Number of Night Awakenings', 'EPDS_SCORE': 'EPDS_SCORE'})
    x_vals = filtered_df_translate['night_awakening_number_bb1']
    y_vals = filtered_df_translate['EPDS_SCORE']
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    fig2.add_traces(px.line(x=x_vals, y=p(x_vals), labels={'x': 'Number of Night Awakenings', 'y': 'EPDS_SCORE'}).data)
    st.plotly_chart(fig2, use_container_width=True)

    # Education vs EPDS_SCORE (simple box plot)
    fig3 = px.box(filtered_df_translate, x='Education', y='EPDS_SCORE',
                  title='EPDS_SCORE vs Education',
                  labels={'Education': 'Education', 'EPDS_SCORE': 'EPDS_SCORE'})
    st.plotly_chart(fig3, use_container_width=True)

    # how_falling_asleep_bb1 vs Sleep_hours
    fig4 = px.box(filtered_df_translate, x='how_falling_asleep_bb1', y='Sleep_hours',
                  title='Sleep Hours vs How Falling Asleep',
                  labels={'how_falling_asleep_bb1': 'How Falling Asleep', 'Sleep_hours': 'Sleep Hours'})
    st.plotly_chart(fig4, use_container_width=True)

    # Explanations about the observed relationships
    st.subheader("Interpretation of Relationships")
    st.markdown("""
    - **Sleep_hours**: A negative relationship is observed between sleep hours and EPDS score, indicating that more sleep hours are associated with lower postpartum depression scores.
    - **night_awakening_number_bb1**: There is a trend of increasing EPDS scores with the increase in the number of night awakenings, suggesting that more night awakenings may be associated with higher levels of postpartum depression.
    - **Education**: Higher levels of education seem to be associated with lower EPDS scores, indicating that education may have a protective effect against postpartum depression.
    """)

with tab3:
    cols = [
        'EPDS_SCORE', 'HADS_SCORE', 'CBTS_SCORE', 'Sleep_hours', 'Age_bb',
        'night_awakening_number_bb1', 'how_falling_asleep_bb1', 'Marital_status_edit',
        'Gestationnal_age', 'Age', 'Education', 'sex_baby1', 'Type_pregnancy'
    ]

    kendall_correlations, _ = calculate_kendall_correlations(filtered_df_translate, cols)
    spearman_correlations, _ = calculate_correlations(filtered_df_translate, cols)

    st.subheader("Correlation Map (Kendall)")
    plot_kendall_heatmap(kendall_correlations)

    st.subheader("Correlation Map (Spearman)")
    plot_spearman_heatmap(spearman_correlations)

    st.subheader("Multiple Linear Regression Analysis")
    perform_multiple_regression(df)

with tab4:
    st.title("Normality Tests")

    # Perform normality tests
    normality_results, summary_df = normality_tests(df, cols)

    # Display summary table
    st.subheader("Summary of Normality Tests")
    st.dataframe(summary_df)

    # Display detailed results
    for col, tests in normality_results.items():
        st.subheader(f"Results for {col}")
        st.write(f"**Shapiro-Wilk**: Statistic={tests['Shapiro-Wilk'][0]:.4f}, p-value={tests['Shapiro-Wilk'][1]:.4f}")
        st.write(f"**Kolmogorov-Smirnov**: Statistic={tests['Kolmogorov-Smirnov'][0]:.4f}, p-value={tests['Kolmogorov-Smirnov'][1]:.4f}")
        st.write(f"**Anderson-Darling**: Statistic={tests['Anderson-Darling'].statistic:.4f}, Critical Values={tests['Anderson-Darling'].critical_values}")

        # Interpretation of results
        st.write("### Interpretation")
        if tests['Shapiro-Wilk'][1] < 0.05:
            st.write("- **Shapiro-Wilk**: The data is not normally distributed (p-value < 0.05).")
        else:
            st.write("- **Shapiro-Wilk**: The data is normally distributed (p-value >= 0.05).")

        if tests['Kolmogorov-Smirnov'][1] < 0.05:
            st.write("- **Kolmogorov-Smirnov**: The data is not normally distributed (p-value < 0.05).")
        else:
            st.write("- **Kolmogorov-Smirnov**: The data is normally distributed (p-value >= 0.05).")

        if tests['Anderson-Darling'].statistic > tests['Anderson-Darling'].critical_values[2]:
            st.write("- **Anderson-Darling**: The data is not normally distributed (statistic > critical value for 5%).")
        else:
            st.write("- **Anderson-Darling**: The data is normally distributed (statistic <= critical value for 5%).")

        # Graphical visualizations
        data = df[col].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(data, kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram of {col}')
        sm.qqplot(data, line='s', ax=axes[1])
        axes[1].set_title(f'Q-Q Plot of {col}')
        st.pyplot(fig)