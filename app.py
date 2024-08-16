import pandas as pd
import os
import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Initialize the Gemini API client
genai.configure(api_key="YOUR_API_KEY")  # Replace with your actual API key

def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

def generate_description(df):
    # Extract column names and basic stats to create a more accurate prompt
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Constructing a prompt based on the structure of the CSV file
    prompt = "This dataset contains the following columns:\n\n"
    
    if num_cols.size > 0:
        prompt += "Numerical columns: " + ", ".join(num_cols) + ".\n"
        prompt += "These columns contain numerical data that can be used for statistical analysis or visualizations.\n"
    
    if cat_cols.size > 0:
        prompt += "Categorical columns: " + ", ".join(cat_cols) + ".\n"
        prompt += "These columns contain categorical data that can be used for classification or grouping.\n"
    
    prompt += "\nDescribe the overall content and potential insights from this data."

    # Generate text using Gemini based on the constructed prompt
    response = genai.generate_text(prompt=prompt)
    return response.result

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

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### Data Preview:")
    st.write(df)
    
    if not df.empty:
        # Generate an initial description of the CSV file
        st.write("### Initial Data Description:")
        initial_description = generate_description(df)
        st.write(initial_description)

        # Provide hints for user queries
        st.write("### Suggested Questions:")
        hints = generate_hint(df)
        for hint in hints:
            st.write(f"- {hint}")
    
        # Initialize session state to store conversation history
        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        # Chat interface for user queries
        with st.form(key="chat_form"):
            query = st.text_input("🗣️ Ask a question or request a visualization")
            submit_button = st.form_submit_button(label="Send")

        # Process the query and store the conversation
        if submit_button and query:
            with st.spinner("Generating response..."):
                description = generate_description(query)
                st.session_state.conversation.append({"user": query, "ai": description})
        
        # Display the conversation history
        for message in st.session_state.conversation:
            st.write(f"**User:** {message['user']}")
            st.write(f"**AI:** {message['ai']}")
        
        # Generate and display a plot based on the user's last query
        if len(st.session_state.conversation) > 0:
            plot_type = st.selectbox("Choose plot type", ["scatter", "line", "bar", "histogram"])
            x_col = st.selectbox("Select X column", options=[None] + df.columns.tolist())
            y_col = st.selectbox("Select Y column", options=[None] + df.columns.tolist())
            
            plot_image = create_plot(df, plot_type, x_col, y_col)
            if plot_image:
                st.image(plot_image)
