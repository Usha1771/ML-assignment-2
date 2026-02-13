import streamlit as st
import pandas as pd
import altair as alt

# Load results
df = pd.read_csv("results_summary.csv")

st.title("Model Performance Comparison")

st.markdown("This app shows Accuracy, AUC, Precision, Recall, F1 Score, and MCC for all models.")

# Display table with formatted floats
st.subheader("Results Table")
st.dataframe(df.style.format({
    "Accuracy": "{:.4f}",
    "AUC": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}",
    "F1 Score": "{:.4f}",
    "MCC": "{:.4f}"
}))

# Dropdown to select metric
metric = st.selectbox(
    "Select a metric to visualize:",
    ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
)

# Bar chart with fixed axis scaling
st.subheader(f"{metric} by Model")
chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("Model", sort="-y"),
    y=alt.Y(metric, scale=alt.Scale(domain=[0, 1])),
    tooltip=["Model", metric]
).properties(width=600)

st.altair_chart(chart, use_container_width=True)

# Highlight best model with formatted value
best_model = df.loc[df[metric].idxmax()]
st.markdown(f"**Best {metric}:** {best_model['Model']} ({best_model[metric]:.4f})")
