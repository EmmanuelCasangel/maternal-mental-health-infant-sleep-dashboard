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



def create_df_brazil(df_translate):
    """
    Create a DataFrame for Brazil's education levels based on predefined probabilities.

    Args:
        df_translate (pd.DataFrame): The original dataset to determine the sample size.

    Returns:
        pd.DataFrame: A DataFrame with simulated education levels for Brazil.
    """
    # Define the probabilities for each education level
    probabilities = [0.041, 0.475, 0.365, 0.119]

    # Set sample_size to match the size of the original dataset
    sample_size = len(df_translate)

    # Calculate the exact number of occurrences for each level
    counts = [int(sample_size * p) for p in probabilities]

    # Adjust to ensure the sum matches the sample size
    while sum(counts) < sample_size:
        counts[counts.index(max(counts))] += 1
    while sum(counts) > sample_size:
        counts[counts.index(max(counts))] -= 1

    # Create the list of education levels with exact proportions
    education_levels = (
        [1] * counts[0] +
        [2] * counts[1] +
        [3] * counts[2] +
        [4] * counts[3]
    )

    # Shuffle the values to distribute them randomly
    np.random.shuffle(education_levels)

    # Create the DataFrame
    df_brazil = pd.DataFrame({'EducationBrazil': education_levels})

    # Mapear os valores para os rótulos descritivos
    education_labels = {
        1: "Sem escolaridade",
        2: "Ensino fundamental",
        3: "Ensino médio",
        4: "Ensino superior"
    }
    df_brazil['EducationBrazil_Labels'] = df_brazil['EducationBrazil'].map(education_labels)

    return df_brazil


# Streamlit Interface
st.set_page_config(page_title="Maternal and Infant Health Dashboard", layout="wide")

# Load data
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Preprocessing
df = preprocess(df)
df_translate = translate_values(df.copy())

# Create DataFrame for Brazil's education levels
df_brazil = create_df_brazil(df_translate)


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

# Adicionar filtro no sidebar para selecionar níveis de escolaridade
selected_education_brazil = st.sidebar.multiselect(
    "Education (Brazil Simulation)",
    options=df_brazil['EducationBrazil_Labels'].unique(),
    default=df_brazil['EducationBrazil_Labels'].unique()
)

# Apply filters
filtered_df_translate = df_translate[
    (df_translate['Age'].between(age_range[0], age_range[1])) &
    (df_translate['Education'].isin(selected_education))
]

# Filtrar o DataFrame com base na seleção
filtered_df_brazil = df_brazil[df_brazil['EducationBrazil_Labels'].isin(selected_education_brazil)]

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
tab1, tab2, tab3, tab4 = st.tabs(["Descriptive Analysis", "Factors Associated with Postpartum Depression (EPDS_SCORE)", "Simulation", "Analyses and Tool"])

with tab1:
    st.header("Distribution of Maternal Age")

    # Pie chart with age ranges, percentages, and custom legend
    bins_age = [19, 26, 31, 36, 48]
    labels_age = ['19-25', '26-30', '31-35', '36-47']
    filtered_df_translate['Age_Category'] = pd.cut(filtered_df_translate['Age'], bins=bins_age, labels=labels_age, right=False)
    age_counts = filtered_df_translate['Age_Category'].value_counts()
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
    education_counts = filtered_df_translate['Education'].value_counts().sort_index()

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
    marital_status_counts = filtered_df_translate['Marital_status_edit'].value_counts()

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
    category_counts = filtered_df_translate['Age_bb'].value_counts().sort_index()

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
    awake_counts = filtered_df_translate['night_awakening_number_bb1'].value_counts().sort_index()

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
    hads_category_counts = filtered_df_translate['HADS_Category'].value_counts().reset_index()
    hads_category_counts.columns = ['HADS_Category', 'count']

    fig = px.bar(hads_category_counts,
                 x='HADS_Category', y='count',
                 color='HADS_Category',
                 color_discrete_sequence=chart_colors,
                 labels={'HADS_Category': 'Category', 'count': 'Count'},
                 title='Distribution of HADS Categories')
    st.plotly_chart(fig, use_container_width=True)

    st.header("Distribution of EPDS Scores")

    # Bar chart for EPDS scores
    epds_score_counts = filtered_df_translate['EPDS_SCORE'].value_counts().reset_index()
    epds_score_counts.columns = ['EPDS_SCORE', 'count']

    fig_epds = px.bar(
        epds_score_counts,
        x='EPDS_SCORE',
        y='count',
        color='EPDS_SCORE',
        color_continuous_scale=chart_colors,  # Escala contínua
        labels={'EPDS_SCORE': 'EPDS Score', 'count': 'Count'},
        title='Distribution of EPDS Scores'
    )

    # Adicionar linha de referência no valor 12
    fig_epds.add_shape(
        type='line',
        x0=12, x1=12, y0=0, y1=epds_score_counts['count'].max(),
        line=dict(color='Red', width=2, dash='dash')
    )

    # Adicionar anotação para a linha de referência
    fig_epds.add_annotation(
        x=12, y=epds_score_counts['count'].max(),
        text='EPDS Score = 12',
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

    # Calculate percentages
    total_count = epds_score_counts['count'].sum()
    below_12_count = epds_score_counts[epds_score_counts['EPDS_SCORE'] < 12]['count'].sum()
    above_12_count = epds_score_counts[epds_score_counts['EPDS_SCORE'] >= 12]['count'].sum()

    below_12_percentage = (below_12_count / total_count) * 100
    above_12_percentage = (above_12_count / total_count) * 100

    # Add annotations for percentages
    fig_epds.add_annotation(
        x=6, y=epds_score_counts['count'].max() * 0.9,
        text=f'Below 12: {below_12_percentage:.1f}%',
        showarrow=False,
        font=dict(size=12, color='Green')
    )

    fig_epds.add_annotation(
        x=18, y=epds_score_counts['count'].max() * 0.9,
        text=f'Above 12: {above_12_percentage:.1f}%',
        showarrow=False,
        font=dict(size=12, color='Red')
    )

    st.plotly_chart(fig_epds, use_container_width=True)


with tab2:
    st.title("Factors Associated with Postpartum Depression (EPDS_SCORE)")

    # Calculate correlations
    correlation_sleep_hours = filtered_df_translate['EPDS_SCORE'].corr(filtered_df_translate['Sleep_hours'],
                                                                       method='spearman')
    correlation_night_awakenings = filtered_df_translate['EPDS_SCORE'].corr(
        filtered_df_translate['night_awakening_number_bb1'], method='spearman')

    # Interactive box plots with trend line and correlation index
    st.subheader("Relationship between EPDS_SCORE and Independent Variables")

    # Sleep_hours vs EPDS_SCORE
    fig1 = px.box(filtered_df_translate, x='Sleep_hours', y='EPDS_SCORE',
                  title=f'EPDS_SCORE vs Sleep_hours (Correlation: {correlation_sleep_hours:.2f})',
                  labels={'Sleep_hours': 'Sleep Hours', 'EPDS_SCORE': 'EPDS_SCORE'})
    x_vals = filtered_df_translate['Sleep_hours']
    y_vals = filtered_df_translate['EPDS_SCORE']
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    fig1.add_traces(px.line(x=x_vals, y=p(x_vals), labels={'x': 'Sleep Hours', 'y': 'EPDS_SCORE'}).data)
    st.plotly_chart(fig1, use_container_width=True)

    # night_awakening_number_bb1 vs EPDS_SCORE
    fig2 = px.box(filtered_df_translate, x='night_awakening_number_bb1', y='EPDS_SCORE',
                  title=f'EPDS_SCORE vs Number of Night Awakenings (Correlation: {correlation_night_awakenings:.2f})',
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
    # # Exibir o grafico da escolaridade no brazil # #

    # Section: Distribution of Education Level and Brazilian Education Level
    st.header("Distribution of Education Levels")

    # Chart 1: Original Education Distribution
    st.subheader("Original Education Distribution")
    education_counts = filtered_df_translate['Education'].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bars1 = ax1.bar(education_counts.index, education_counts.values, color=chart_colors)

    ax1.set_xticks(education_counts.index)
    ax1.set_xticklabels(education_counts.index, rotation=0, ha='center')

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    ax1.set_xlabel('Education')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='best')

    plt.tight_layout()
    st.pyplot(fig1)


    #############################################################################################################################################

    # Original Distribution of EPDS Scores
    st.header("Original Distribution of EPDS Scores")

    # Bar chart for original EPDS scores
    epds_score_counts_original = filtered_df_translate['EPDS_SCORE'].value_counts().reset_index()
    epds_score_counts_original.columns = ['EPDS_SCORE', 'count']

    fig_epds_original = px.bar(
        epds_score_counts_original,
        x='EPDS_SCORE',
        y='count',
        color='EPDS_SCORE',
        color_continuous_scale=chart_colors,  # Escala contínua
        labels={'EPDS_SCORE': 'EPDS Score', 'count': 'Count'},
        title='Original Distribution of EPDS Scores'
    )

    # Add reference line at value 12
    fig_epds_original.add_shape(
        type='line',
        x0=12, x1=12, y0=0, y1=epds_score_counts_original['count'].max(),
        line=dict(color='Red', width=2, dash='dash')
    )

    # Add annotation for the reference line
    fig_epds_original.add_annotation(
        x=12, y=epds_score_counts_original['count'].max(),
        text='EPDS Score = 12',
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

    # Calculate percentages
    total_count_original = epds_score_counts_original['count'].sum()
    below_12_count_original = epds_score_counts_original[epds_score_counts_original['EPDS_SCORE'] < 12]['count'].sum()
    above_12_count_original = epds_score_counts_original[epds_score_counts_original['EPDS_SCORE'] >= 12]['count'].sum()

    below_12_percentage_original = (below_12_count_original / total_count_original) * 100
    above_12_percentage_original = (above_12_count_original / total_count_original) * 100

    # Add annotations for percentages
    fig_epds_original.add_annotation(
        x=6, y=epds_score_counts_original['count'].max() * 0.9,
        text=f'Below 12: {below_12_percentage_original:.1f}%',
        showarrow=False,
        font=dict(size=12, color='Green')
    )

    fig_epds_original.add_annotation(
        x=18, y=epds_score_counts_original['count'].max() * 0.9,
        text=f'Above 12: {above_12_percentage_original:.1f}%',
        showarrow=False,
        font=dict(size=12, color='Red')
    )

    st.plotly_chart(fig_epds_original, use_container_width=True)




    ####################################################################################################################################


    # Chart 2: Brazilian Education Distribution
    st.subheader("Brazilian Education Distribution")
    brazil_education_counts = filtered_df_brazil['EducationBrazil_Labels'].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bars1 = ax1.bar(brazil_education_counts.index, brazil_education_counts.values, color=chart_colors)

    ax1.set_xticks(brazil_education_counts.index)
    ax1.set_xticklabels(brazil_education_counts.index, rotation=0, ha='center')

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    ax1.set_xlabel('Education')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='best')

    plt.tight_layout()
    st.pyplot(fig1)




with tab4:
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


