import streamlit as st
import pandas as pd
import altair as alt

# Load results
df = pd.read_csv("results_summary.csv")

st.title("Model Performance Comparison")

st.markdown("This app shows Accuracy, AUC, Precision, Recall, F1 Score, and MCC for all models.")

# Display table
st.subheader("Results Table")
st.dataframe(df)

# Dropdown to select metric
metric = st.selectbox(
    "Select a metric to visualize:",
    ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
)

# Bar chart
st.subheader(f"{metric} by Model")
chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("Model", sort="-y"),
    y=metric,
    tooltip=["Model", metric]
).properties(width=600)

st.altair_chart(chart, use_container_width=True)

# Highlight best model
best_model = df.loc[df[metric].idxmax()]
st.markdown(f"**Best {metric}:** {best_model['Model']} ({best_model[metric]:.4f})")
