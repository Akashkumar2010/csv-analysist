import pandas as pd
import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Initialize the Gemini API client
genai.configure(api_key="AIzaSyDFqr07uAzPAB2ahk2ZmnahwX36x1E8gIA")

def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

def generate_description(query):
    # Generate text using Gemini
    response = genai.generate_text(prompt=query)
    return response.result

def suggest_explorations(df):
    suggestions = []
    
    if df is not None:
        # Check for numerical and categorical columns
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        # Basic exploration suggestions
        if num_cols.size > 0:
            suggestions.append("Visualize distributions of numerical columns using histograms or box plots.")
            suggestions.append("Check for correlations between numerical features using a heatmap.")
        
        if cat_cols.size > 0:
            suggestions.append("Explore the distribution of categorical features with bar plots.")
            suggestions.append("Analyze the relationship between categorical and numerical features with group-by operations.")

        if num_cols.size > 1:
            suggestions.append("Scatter plots can help explore relationships between numerical columns.")

        if cat_cols.size > 0 and num_cols.size > 0:
            suggestions.append("Create bar plots to compare categories across numerical values.")

        # Additional advanced suggestions
        if num_cols.size > 0 and df[num_cols].isnull().any().any():
            suggestions.append("Consider handling missing values in numerical columns.")
        
        if cat_cols.size > 0 and df[cat_cols].isnull().any().any():
            suggestions.append("Consider handling missing values in categorical columns.")
        
        if df.duplicated().sum() > 0:
            suggestions.append(f"Found {df.duplicated().sum()} duplicated rows, consider removing them.")
        
        # Unique value checks
        for col in df.columns:
            if df[col].nunique() == df.shape[0]:
                suggestions.append(f"The column '{col}' has unique values for all rows, it might be an identifier.")
        
    return suggestions

def create_plot(df, plot_type, x_col=None, y_col=None):
    plt.figure(figsize=(10, 6))
    
    if plot_type == "scatter" and x_col and y_col:
        sns.scatterplot(data=df, x=x_col, y=y_col)
    elif plot_type == "line" and x_col and y_col:
        sns.lineplot(data=df, x=x_col, y=y_col)
    elif plot_type == "bar" and x_col and y_col:
        df.groupby(x_col)[y_col].mean().plot(kind="bar")
    elif plot_type == "histogram" and x_col:
        df[x_col].plot(kind="hist")
    else:
        st.write("Please specify the required columns for plotting.")
        return None

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

# Streamlit app title
st.title("Professional Data Analysis Dashboard")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV data", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader("Data Preview")
    st.write(df)
    
    if not df.empty:
        # Data summary
        st.subheader("Data Summary")
        st.write(df.describe())
        
        # Initial data description
        initial_query = "Provide a summary description for this dataset."
        initial_description = generate_description(initial_query)
        st.subheader("Initial Data Description")
        st.write(initial_description)

        # Automatically generated exploration suggestions
        st.subheader("Suggested Explorations")
        suggestions = suggest_explorations(df)
        for suggestion in suggestions:
            st.write(f"- {suggestion}")

        # Chat interface for queries and visualizations
        st.subheader("Chat with Your Data")
        chat_input = st.text_area("Ask a question or request a visualization:")
        plot_type = st.selectbox("Choose plot type", ["scatter", "line", "bar", "histogram"])
        x_col = st.selectbox("Select X column", options=[None] + df.columns.tolist())
        y_col = st.selectbox("Select Y column", options=[None] + df.columns.tolist())
        chat_button = st.button("Generate Response")

        if chat_button and chat_input:
            with st.spinner("Processing..."):
                chat_description = generate_description(chat_input)
                st.subheader("AI-Generated Response")
                st.write(chat_description)

                # Generate and display plot
                st.subheader("Generated Visualization")
                plot_image = create_plot(df, plot_type, x_col, y_col)
                if plot_image:
                    st.image(plot_image)
