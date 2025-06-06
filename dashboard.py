import pandas as pd
import streamlit as st
import plotly.express as px
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import itertools
import plotly.graph_objects as go

from preprocessamento import preprocess, translate_values
from correlation_kendall_heatmap import calculate_kendall_correlations, plot_heatmap as plot_kendall_heatmap
from correlation_spearman_heatmap import calculate_correlations, plot_heatmap as plot_spearman_heatmap
from multiple_regression import perform_multiple_regression

# Criar uma nova coluna com valores aleatórios baseados na média e desvio padrão
def generate_epds_scores(row, stats):
    # Filter the mean and standard deviation for the corresponding education category
    education_row = stats[stats['Education'] == row['EducationBrazil']]
    mean = education_row['Mean_EPDS_SCORE'].values[0]
    std = education_row['Std_EPDS_SCORE'].values[0]
    # Generate a random value based on the mean and standard deviation, ensuring it is between 0 and 30, and round to an integer
    return int(round(min(30, max(0, np.random.normal(loc=mean, scale=std)))))

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


# Charts
st.subheader("")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Descriptive Analysis", "Factors Associated with Postpartum Depression (EPDS_SCORE)", "Clusters", "Simulation", "Analyses and Tool"])

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
        color_continuous_scale=chart_colors[::-1],  # Escala contínua
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
    # Obter todas as colunas do DataFrame
    todas_colunas = df.columns.tolist()

    # Definir as colunas padrão
    colunas_padrao = ['how_falling_asleep_bb1', 'Sleep_hours']

    # Criar um seletor interativo no Streamlit
    colunas_interesse = st.multiselect(
        "Selecione as colunas de interesse:",
        options=todas_colunas,  # Todas as colunas disponíveis
        default=colunas_padrao  # Colunas padrão selecionadas inicialmente
    )

    # Filtrar o DataFrame
    df_cluster = df[colunas_interesse].dropna()


    # Normalizar os dados
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df_cluster)

    # Método do cotovelo
    inertia = []
    k_range = range(1, 11)  # Testar de 1 a 10 clusters

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_normalized)
        inertia.append(kmeans.inertia_)

    # Plot the elbow method graph
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')
    plt.grid()

    # Use st.pyplot() to display the plot in Streamlit
    st.pyplot(plt)


    # Calculate Silhouette Scores for different numbers of clusters
    silhouette_scores = []

    for k in range(2, 11):  # Silhouette Score requires at least 2 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(df_normalized)
        score = silhouette_score(df_normalized, cluster_labels)
        silhouette_scores.append(score)

    # Plot the Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.grid()
    st.pyplot(plt)


    ############ Analise do numero de componentes ideal PCAS #############################

    ########## Calcular a variância explicada ##########
    # Definir o número de componentes principais como o máximo possível
    num_components = df_normalized.shape[1]

    # Aplicar PCA com o número selecionado de componentes
    pca_analise = PCA(n_components=num_components)
    pca_result = pca_analise.fit_transform(df_normalized)

    # Obter a variância explicada por cada componente
    explained_variance = pca_analise.explained_variance_ratio_

    # Calcular o acumulado da variância explicada
    cumulative_variance = explained_variance.cumsum()

    # Visualizar a variância explicada e acumulada em um gráfico de barras com linha
    plt.figure(figsize=(10, 6))

    # Gráfico de barras para a variância explicada
    plt.bar(range(1, num_components + 1), explained_variance, alpha=0.7, label='Explained Variance', color='skyblue')

    # Linha para a variância acumulada
    plt.plot(range(1, num_components + 1), cumulative_variance, marker='o', color='orange', label='Cumulative Variance')

    # Adicionar rótulos e título
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Ratio')
    plt.title('Explained and Cumulative Variance by PCA')
    plt.xticks(range(1, num_components + 1))  # Ajustar os ticks do eixo x
    plt.legend()
    plt.grid()

    # Exibir o gráfico
    st.pyplot(plt)


    ########################################################################################

    ####################### escolher k = 3 #######################
    # Fit KMeans with k
    # Add a slider to select the number of clusters (k)
    k = st.slider("Select the number of clusters (k):", min_value=2, max_value=10, value=5, step=1)

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(df_normalized)

    # Add cluster labels to the DataFrame
    df_cluster['Cluster'] = cluster_labels


    ###########

    # Adicionar um expander para os gráficos de clusters sem uso de PCAS
    with st.expander("Visualizar Gráficos de Clusters, sem PCAS", expanded=False):
        # Iterar sobre todas as combinações de pares de colunas selecionadas
        for col_x, col_y in itertools.combinations(colunas_interesse, 2):
            plt.figure(figsize=(8, 6))

            # Criar gráfico de dispersão
            plt.scatter(
                df_cluster[col_x],
                df_cluster[col_y],
                c=df_cluster['Cluster'],
                cmap='viridis',
                s=50,
                alpha=0.7
            )

            # Adicionar rótulos e título
            plt.xlabel(col_x)
            plt.ylabel(col_y)
            plt.title(f'Clusters: {col_x} vs {col_y}')
            plt.colorbar(label='Cluster')
            plt.grid()

            # Exibir o gráfico
            st.pyplot(plt)

    ############


    ############### Uso de PCAs ######################

    # Adicionar um controle deslizante para selecionar o número de componentes principais
    num_components = st.slider(
        "Selecione o número de componentes principais (PCA):",
        min_value=2,
        max_value=3,
        value=2,  # Valor inicial
        step=1
    )

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=num_components)
    df_pca = pca.fit_transform(df_normalized)

    # Fit KMeans with k on the PCA-reduced data
    kmeans_pca = KMeans(n_clusters=k, random_state=42)
    cluster_labels_pca = kmeans_pca.fit_predict(df_pca)

    # Visualize the clusters with PCA and add variable vectors
    if df_pca.shape[1] == 2:  # 2D plot for 2 PCAs
        plt.figure(figsize=(8, 5))
        plt.scatter(df_pca[:, 0], df_pca[:, 1], c=cluster_labels_pca, cmap='viridis', s=50)
        plt.title(f'Clusters Visualization with PCA (k={k})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid()

        # Add variable vectors (loadings)
        for i, (x, y) in enumerate(pca.components_.T):  # Transpose to iterate over variables
            plt.arrow(0, 0, x, y, color='red', alpha=0.5, head_width=0.05)
            plt.text(x * 1.1, y * 1.1, df_cluster.columns[i], color='red', fontsize=10)

        st.pyplot(plt)

    elif df_pca.shape[1] == 3:  # 3D plot for 3 PCAs

        fig = go.Figure()

        # Add scatter points for clusters
        fig.add_trace(go.Scatter3d(
            x=df_pca[:, 0],
            y=df_pca[:, 1],
            z=df_pca[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=cluster_labels_pca,  # Color by cluster
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Clusters'
        ))

        # Add variable vectors (loadings) with arrows
        for i, (x, y, z) in enumerate(pca.components_.T):  # Transpose to iterate over variables
            # Add a line for the vector
            fig.add_trace(go.Scatter3d(
                x=[0, x],
                y=[0, y],
                z=[0, z],
                mode='lines',
                line=dict(color='red', width=4),
                name=f'Variable: {df_cluster.columns[i]}'
            ))
            # Add a cone for the arrowhead
            fig.add_trace(go.Cone(
                x=[x], y=[y], z=[z],
                u=[x], v=[y], w=[z],
                colorscale=[[0, 'red'], [1, 'red']],
                sizemode="absolute",
                sizeref=0.2,  # Adjust the size of the arrowhead
                anchor="tip",
                showscale=False
            ))

        # Update layout for better visualization
        fig.update_layout(
            title=f'Clusters Visualization with PCA (k={k})',
            scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Principal Component 3'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Display the interactive plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)


    ########### Obter os pesos (loadings) dos PCAs ###########
    loadings = pd.DataFrame(pca.components_, columns=df_cluster.columns[:-1], index=[f'PCA{i+1}' for i in range(pca.n_components_)])

    # Exibir os pesos
    print(loadings)

    # Visualizar os pesos como um heatmap
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Pesos (Loadings) dos PCAs')
    st.pyplot(plt)


    ########## Relacionar os PCAs com as Variáveis Originais ##########
    # Adicionar os PCAs ao DataFrame original
    pca_df = pd.DataFrame(df_pca, columns=[f'PCA{i + 1}' for i in range(pca.n_components_)])
    pca_df = pd.concat([pca_df, df_cluster.reset_index(drop=True)], axis=1)

    # Calcular correlações
    correlations = pca_df.corr()

    # Visualizar as correlações dos PCAs com as variáveis originais
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations.iloc[:pca.n_components_, pca.n_components_:], annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlação entre PCAs e Variáveis Originais')
    st.pyplot(plt)

with tab4:
    # # Exibir o grafico da escolaridade no brazil # #

    # Section: Distribution of Education Level and Brazilian Education Level
    st.header("Distribution of Education Levels")

    # Chart 1: Original Education Distribution
    st.subheader("Original Education Distribution")
    education_counts = filtered_df_translate['Education'].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(12, 8))
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
        color_continuous_scale=chart_colors[::-1],  # Escala contínua
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

    fig1, ax1 = plt.subplots(figsize=(12, 8))
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

    ############################################################################################################################################################################################################

    # Combine categories 4 and 5 into a single category
    df['Education_University_Grouped'] = df['Education'].replace(
        5, 4)

    # Calculate mean and standard deviation for each education category
    education_stats = df.groupby('Education_University_Grouped')['EPDS_SCORE'].agg(
        ['mean', 'std']).reset_index()

    # Rename columns for clarity
    education_stats.columns = ['Education', 'Mean_EPDS_SCORE', 'Std_EPDS_SCORE']

    # Aplicar a função para gerar os valores
    filtered_df_brazil['EPDS_SCORE_BRAZIL'] = filtered_df_brazil.apply(
        generate_epds_scores, axis=1, stats=education_stats
    )

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    # Simulated Distribution of EPDS Scores for Brazil
    st.header("Simulated Distribution of EPDS Scores for Brazil")

    # Count the frequency of EPDS_SCORE_BRAZIL
    epds_score_counts_brazil = filtered_df_brazil['EPDS_SCORE_BRAZIL'].value_counts().reset_index()
    epds_score_counts_brazil.columns = ['EPDS_SCORE_BRAZIL', 'count']

    # Create the bar chart
    fig_epds_brazil = px.bar(
        epds_score_counts_brazil,
        x='EPDS_SCORE_BRAZIL',
        y='count',
        color='EPDS_SCORE_BRAZIL',
        color_continuous_scale=chart_colors[::-1],  # Reverse color scale
        labels={'EPDS_SCORE_BRAZIL': 'EPDS Score (Brazil)', 'count': 'Count'},
        title='Simulated Distribution of EPDS Scores for Brazil'
    )

    # Add reference line at value 12
    fig_epds_brazil.add_shape(
        type='line',
        x0=12, x1=12, y0=0, y1=epds_score_counts_brazil['count'].max(),
        line=dict(color='Red', width=2, dash='dash')
    )

    # Add annotation for the reference line
    fig_epds_brazil.add_annotation(
        x=12, y=epds_score_counts_brazil['count'].max(),
        text='EPDS Score = 12',
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

    # Calculate percentages
    total_count_brazil = epds_score_counts_brazil['count'].sum()
    below_12_count_brazil = epds_score_counts_brazil[epds_score_counts_brazil['EPDS_SCORE_BRAZIL'] < 12]['count'].sum()
    above_12_count_brazil = epds_score_counts_brazil[epds_score_counts_brazil['EPDS_SCORE_BRAZIL'] >= 12]['count'].sum()

    below_12_percentage_brazil = (below_12_count_brazil / total_count_brazil) * 100
    above_12_percentage_brazil = (above_12_count_brazil / total_count_brazil) * 100

    # Add annotations for percentages
    fig_epds_brazil.add_annotation(
        x=6, y=epds_score_counts_brazil['count'].max() * 0.9,
        text=f'Below 12: {below_12_percentage_brazil:.1f}%',
        showarrow=False,
        font=dict(size=12, color='Green')
    )

    fig_epds_brazil.add_annotation(
        x=18, y=epds_score_counts_brazil['count'].max() * 0.9,
        text=f'Above 12: {above_12_percentage_brazil:.1f}%',
        showarrow=False,
        font=dict(size=12, color='Red')
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig_epds_brazil, use_container_width=True)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    # Map for Brazilian Labels
    brazilian_labels_map = {
        1: "Sem escolaridade",
        2: "Ensino fundamental",
        3: "Ensino médio",
        4: "Ensino superior"
    }

    # Map for Original Labels
    original_labels_map = {
        1: "No education",
        2: "Compulsory school",
        3: "Post-compulsory education",
        4: "UAS or UT Degree + University",
    }

    # Add the new columns to the DataFrame
    education_stats['Brazilian Labels'] = education_stats['Education'].map(brazilian_labels_map)
    education_stats['Original Labels'] = education_stats['Education'].map(original_labels_map)

    # Display the updated DataFrame
    st.write(education_stats)

    ############################################################################################################################################################################################################xwww

with tab5:
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


