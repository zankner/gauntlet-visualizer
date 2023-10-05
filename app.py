import streamlit as st
from io import StringIO
import pandas as pd
from typing import List, Tuple

def grab_categories_and_values(df: pd.DataFrame, model_name: str = None) -> Tuple[List[str], List[float]]:

    categories =  list(df.loc[model_name].index)
    values = list(df.loc[model_name].values)

    # append the first category to the end to close the loop
    categories.append(categories[0])
    values.append(values[0])

    return categories, values

st.title("LLM evaluator")

# Add a sidebar title
st.sidebar.title('Eval Markdown Input')

md_default = """
| model_name               |   average |   world_knowledge |   commonsense_reasoning |   language_understanding |   symbolic_problem_solving |   reading_comprehension |
|:-------------------------|----------:|------------------:|------------------------:|-------------------------:|---------------------------:|------------------------:|
"""
# Create a text area in the sidebar
md_table_string = st.sidebar.text_area("Paste your markdown text here (no triple quotes)", md_default, height=500)

# st.write(md_table_string)

print(md_table_string)

md_df = pd.read_csv(
    StringIO(md_table_string.replace(' ', '')),  # Get rid of whitespaces
    sep='|',
    index_col=1
).dropna(
    axis=1,
    how='all'
).iloc[1:]
models = list(md_df.index)
md_df = md_df.reset_index()
md_df = md_df.set_index('model_name')
# convert all columns to float type, except for the index
md_df = md_df.astype(float)

# two columns for the model names and the metrics
col1, col2 = st.columns(2)

with col1:
    # show a multiselect widget with the model names. default to just "mosaicml/mpt-7b"
    selected_models = st.multiselect(
        'Select models to compare',
        list(md_df.index),
        default=[]
    )

with col2:
    # show a multiselect widget with the metrics. default to to everything except for "average"
    selected_metrics = st.multiselect(
        'Select metrics to compare',
        list(md_df.columns),
        default=list(md_df.columns)
    )

# restrict the dataframe to the selected models and metrics
md_df = md_df[selected_metrics].loc[selected_models]

# show the dataframe
st.write(md_df)

import plotly.graph_objects as go

# loop over model names. for each, grab the categories and values and add a trace to the figure
# build a streamlit plotly chart

fig = go.Figure()

for model_name in selected_models:
    categories, values = grab_categories_and_values(md_df, model_name)
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=model_name
    ))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0., 1.]
    )),
  showlegend=True
)

# fig.show()
st.plotly_chart(fig)