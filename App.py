import streamlit as st
import polars as pl

st.image("D64 Logo.png", width=150)

DATA_PATH = "data/issues_postprocessed.parquet"

st.title("D-Stack Feedback Analytics")
st.text("""
Interaktive Analyse von Feedback-Einreichungen, die im Rahmen des Konsultationsprozess zum Deutschland-Stack erstellt wurden.
Labels und Sentiment Scores wurden mit Hilfe von NLP / GenAI Modellen automatisch zugewiesen.""")

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

st.divider()

st.subheader("Stichproben der Issues")

st.button("Neue Stichprobe ziehen")

df = df.filter(
    pl.all_horizontal([pl.col("labels").list.contains(label) for label in st.session_state.label_filter]) if st.session_state.label_filter else pl.lit(True)
).filter(
    pl.col("form_page").is_in(st.session_state.page_filter) if st.session_state.page_filter else pl.lit(True)
).filter(
    (pl.col("sentiment") >= st.session_state.sentiment_filter[0]) & 
    (pl.col("sentiment") <= st.session_state.sentiment_filter[1])
).filter(
    pl.col("author_name").is_in(st.session_state.author_filter) if "author_filter" in st.session_state and st.session_state.author_filter else pl.lit(True)
).filter(
    pl.col("desc_clean").str.to_lowercase().str.contains(st.session_state.search_term.lower()) if "search_term" in st.session_state and st.session_state.search_term else pl.lit(True)
)

sample = df.sample(1) if df.height > 0 else pl.DataFrame(schema=df.schema)
if sample.height > 0:
    row = sample.row(0, named=True)
    st.markdown(f"""
**Titel:** {row['title']}

**Inhalt:** {row['desc_clean']}

**Aus Seite:** {row['form_page']}

**Labels:** {', '.join(row['labels']) if row['labels'] else 'None'}

**Sentiment:** {row['sentiment']:.2f}

**Eingereicht von:** {row['author_name']}
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

st.subheader("Zeitliche Verteilung der Issues")

st.line_chart(
    df.group_by(pl.col("created_at").dt.date())
    .agg(pl.len())
    .sort("created_at")
    .rename({"created_at": "Datum", "len": "Anzahl Issues"})
    .to_pandas()
    .set_index("Datum")
)

col1, col2, col3 = st.columns(3)

st.subheader("Häufigkeit der Labels")
st.bar_chart(
    df.explode("labels").group_by("labels").agg(pl.len()).sort("len", descending=True).rename({"labels": "Label", "len": "Anzahl Issues"})
    .to_pandas().set_index("Label")
    , sort = False
)

st.subheader("Einreichung über Formular erfolgt?")
st.bar_chart(
    df.group_by("is_from_form").agg(pl.len()).sort("len", descending=True).rename({"is_from_form": "Formular", "len": "Anzahl Issues"})
    .to_pandas().set_index("Formular")
    , sort = False
)

st.subheader("Durchschnittlicher Sentiment nach Label")
st.bar_chart(
    df.explode("labels").group_by("labels").agg(pl.col("sentiment").mean()).sort("sentiment", descending=True).rename({"labels": "Label", "sentiment": "Durchschnittlicher Sentiment"})
    .to_pandas().set_index("Label")
    , sort = False
)

st.subheader("Durchschnittlicher Sentiment nach Seite")
st.bar_chart(
    df.group_by("form_page").agg(pl.col("sentiment").mean()).rename({"form_page": "Seite", "sentiment": "Durchschnittlicher Sentiment"})
    .to_pandas().set_index("Seite").sort_values("Durchschnittlicher Sentiment", ascending=False)
    , sort = False
)

st.divider()

st.subheader("Autor*innen der Issues")

st.dataframe(
    df.group_by("author_name").agg(pl.len().alias("Anzahl Issues")).sort("Anzahl Issues", descending=True)
)

st.multiselect(
    "Filter nach Autor*in",
    options=sorted(df["author_name"].unique().to_list()),
    key="author_filter"
)

st.divider()

st.subheader("Stichwortsuche")

st.text("Du suchst nach einem Thema, das nicht über die Labels abgedeckt ist? ")

st.text_input(
    "Suchbegriff",
    key="search_term"
)