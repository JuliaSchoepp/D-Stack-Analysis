import streamlit as st
import polars as pl

DATA_PATH = "data/issues_postprocessed.parquet"

st.title("D-Stack Analytics")
st.header("Analyse der im Konsultationsprozess eingereichten Feedback-Issues")

@st.cache_data
def load_data() -> pl.DataFrame:
    return pl.read_parquet(DATA_PATH)   

df = load_data()

st.multiselect(
    "Filter nach Labels",
    options=df.explode("labels")["labels"].unique().drop_nulls().sort().to_list(),
    key="label_filter"
)

st.multiselect(
    "Filter nach Seite",
    options=df["form_page"].unique().to_list(),
    key="page_filter"
)

st.slider(
    "Sentiment Score Bereich",
    min_value=-1.0,
    max_value=1.0,
    value=(-1.0, 1.0),
    step=0.1,
    key="sentiment_filter"
)

st.subheader("Stichproben der Issues")

st.button("Neue Stichprobe ziehen")

df = df.filter(
    pl.all_horizontal([pl.col("labels").list.contains(label) for label in st.session_state.label_filter]) if st.session_state.label_filter else pl.lit(True)
).filter(
    pl.col("form_page").is_in(st.session_state.page_filter) if st.session_state.page_filter else pl.lit(True)
).filter(
    (pl.col("sentiment") >= st.session_state.sentiment_filter[0]) & 
    (pl.col("sentiment") <= st.session_state.sentiment_filter[1])
)

sample = df.sample(1) if df.height > 0 else pl.DataFrame(schema=df.schema)
if sample.height > 0:
    row = sample.row(0, named=True)
    st.markdown(f"""
**Title:** {row['title']}

**Description:** {row['desc_clean']}

**Form Page:** {row['form_page']}

**Labels:** {', '.join(row['labels']) if row['labels'] else 'None'}

**Sentiment:** {row['sentiment']:.2f}
""")
else:
    st.markdown("No issues match the current filters.")


st.divider()

st.subheader("Übersicht der gefilterten Issues")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Anzahl der Issues", df.height)

with col2:
    st.metric("Durchschnittlicher Sentiment Score", f"{df['sentiment'].mean():.2f}")

with col3:
    st.metric("Anzahl der meldenden Personen", df["author_id"].n_unique())

st.line_chart(
    df.group_by(pl.col("created_at").dt.date()).agg(pl.count()).sort("created_at").rename({"created_at": "Datum", "count": "Anzahl Issues"})
    .to_pandas().set_index("Datum")
)

col1, col2, col3 = st.columns(3)

st.bar_chart(
    df.explode("labels").group_by("labels").agg(pl.count()).sort("count", descending=True).rename({"labels": "Label", "count": "Anzahl Issues"})
    .to_pandas().set_index("Label")
)

st.bar_chart(
    df.group_by("is_from_form").agg(pl.count()).rename({"is_from_form": "Formular", "count": "Anzahl Issues"})
    .to_pandas().set_index("Formular")
)

