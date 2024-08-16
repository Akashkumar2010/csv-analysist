import pandas as pd
import os
import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Initialize the Gemini API client
genai.configure(api_key="AIzaSyDFqr07uAzPAB2ahk2ZmnahwX36x1E8gIA")

def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

def generate_description(query):
    # Generate text using Gemini
    response = genai.generate_text(prompt=query)
    return response.result  # Accessing the correct attribute

def generate_hint(df):
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    hints = []
    
    if num_cols.size > 0:
        hints.append("You can ask about correlations between numerical features.")
        hints.append("Try visualizing the distributions of numerical columns.")
    
    if cat_cols.size > 0:
        hints.append("Explore the distribution of categorical features.")
        hints.append("Analyze how categorical features are related to numerical features.")
    
    if num_cols.size > 1:
        hints.append("Consider creating scatter plots to examine relationships between numerical columns.")
    
    if cat_cols.size > 0 and num_cols.size > 0:
        hints.append("Bar plots can be useful to show comparisons between categories.")
    
    return hints

def create_plot(df, plot_type, x_col=None, y_col=None):
    plt.figure(figsize=(10, 6))
    
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if plot_type == "scatter":
        if x_col and y_col:
            sns.scatterplot(data=df, x=x_col, y=y_col)
        else:
            st.write("Please specify both X and Y columns for a scatter plot.")
            return None
    elif plot_type == "line":
        if x_col and y_col:
            sns.lineplot(data=df, x=x_col, y=y_col)
        else:
            st.write("Please specify both X and Y columns for a line plot.")
            return None
    elif plot_type == "bar":
        if x_col and y_col:
            df.groupby(x_col)[y_col].mean().plot(kind="bar")
        else:
            st.write("Please specify both X and Y columns for a bar plot.")
            return None
    elif plot_type == "histogram":
        if num_cols.size > 0:
            df[num_cols[0]].plot(kind="hist")
        else:
            st.write("No numerical columns available for a histogram.")
            return None
    else:
        st.write("Unsupported plot type!")
        return None

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

# Streamlit app title
st.write("# Chat with CSV Data 🦙")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### Data Preview:")
    st.write(df)
    
    if not df.empty:
        # Provide hints for user queries
        st.write("### Suggested Questions:")
        hints = generate_hint(df)
        for hint in hints:
            st.write(f"- {hint}")
    
        # Chat box to capture user input and display conversation
        query = st.text_input("🗣️ Ask a question or request analysis:", key="query_input")

        if query:
            with st.spinner("Generating response..."):
                # Store user input in chat history
                st.session_state.history.append({"role": "user", "content": query})
                
                # Generate response from AI
                description = generate_description(query)
                st.session_state.history.append({"role": "ai", "content": description})

        # Display chat history
        for message in st.session_state.history:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**AI:** {message['content']}")
        
        # Optional: Generate and display a plot based on the most recent user query
        if st.button("Generate Visualization"):
            plot_type = st.selectbox("Choose plot type", ["scatter", "line", "bar", "histogram"], key="plot_type")
            x_col = st.selectbox("Select X column", options=[None] + df.columns.tolist(), key="x_col")
            y_col = st.selectbox("Select Y column", options=[None] + df.columns.tolist(), key="y_col")
            plot_image = create_plot(df, plot_type, x_col, y_col)
            if plot_image:
                st.image(plot_image)
