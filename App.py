import streamlit as st
import polars as pl
from pathlib import Path
import io
from google.cloud import storage

st.set_page_config(
    page_title="D-Stack Analytics",
    page_icon="D64_Logo.svg"
)

st.image("D64_Logo.svg", width=180)

GCS_BUCKET = "dstack-feedback"
GCS_PREFIX = "data/processing_date="
LABELS_VERSION = 1
LABELS_COLUMN = f"labels_v{LABELS_VERSION}"

st.title("D-Stack Feedback Analytics")
st.markdown("""
### Über das Projekt
Ende November endete die erste Konsultationsphase zum [Deutschland-Stack](https://deutschland-stack.gov.de/), und gerade geht das Projekt in die zweite Feedback-Runde.
Für alle, die keine Lust haben, [alle 500+ Beiträge](https://gitlab.opencode.de/dstack/d-stack-home/-/issues?sort=created_date&state=all&first_page_size=20&show=eyJpaWQiOiI0OTIiLCJmdWxsX3BhdGgiOiJkc3RhY2svZC1zdGFjay1ob21lIiwiaWQiOjQzMjc5fQ%3D%3D) zu lesen, 
sich aber trotzdem einen Eindruck zum eingereichten Feedback verschaffen wollen, haben wir dieses Tool geschaffen.
Das Projekt begann als interne Initiative einer Gruppe von [D64-Mitgliedern](https://d-64.org/), die den Konsultationsprozess mitverfolgt haben und das Feedback besser verstehen wollten.
Wir wollen es allen Interessierten ermöglichen, komfortabel durch das eingereichte Feedback zu navigieren und Einblicke in die Themen und Stimmungen der Beiträge zu gewinnen.

### Methodik
Die hier präsentierten Analysen basieren auf den öffentlich zugänglichen Issues des [GitLab-Projekts zum Deutschland-Stack](https://gitlab.opencode.de/dstack/d-stack-home).
Wir haben alle Issues heruntergeladen und bereinigt.
Darüber hinaus haben wir die Issues mit Hilfe von NLP und GenAI-Modellen um folgende Attribute angereichert:
- **Themen-Labels:** Jedes Issue wurde mit thematischen Labels versehen, um die Kategorisierung zu erleichtern.
- **Sentiment-Analyse:** Jedes Issue wurde auf seine Stimmung hin analysiert, um positive, negative oder neutrale Tendenzen zu identifizieren.
- **Organisationszuordnung:** Wo möglich, wurde versucht, die hinter den Issues stehenden Organisationen zu identifizieren, z.B. wenn diese explizit erwähnt oder in ihrem Namen unterschrieben wurde. 

Die mit Hilfe von KI generierten Attribute können wie immer Fehler und Ungenauigkeiten enthalten.
""")

st.divider()

st.subheader("Filtermöglichkeiten")

@st.cache_data
def load_data() -> pl.DataFrame:
    """Load all partitioned issue data from GCS bucket."""
    try:
        # Use anonymous credentials (bucket is public)
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(GCS_BUCKET)
        
        dfs = []
        for blob in bucket.list_blobs(prefix=GCS_PREFIX):
            if blob.name.endswith("issues.parquet"):
                try:
                    parquet_bytes = blob.download_as_bytes()
                    df = pl.read_parquet(io.BytesIO(parquet_bytes))
                    dfs.append(df)
                except Exception as e:
                    st.warning(f"Failed to load {blob.name}: {e}")
        
        if not dfs:
            st.warning("No data found in GCS bucket")
            return pl.DataFrame()
        
    except Exception as e:
        st.warning(f"GCS connection failed, trying local fallback: {e}")
        # Fallback to local files for development
        data_dir = Path("data/issues_postprocessed")
        parquet_files = list(data_dir.glob("processing_date=*/issues.parquet"))
        if parquet_files:
            dfs = [pl.read_parquet(f) for f in parquet_files]
        else:
            return pl.DataFrame()
    
    # Align schemas across all partitions
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    
    aligned_dfs = []
    for df in dfs:
        missing_cols = all_columns - set(df.columns)
        for col in missing_cols:
            df = df.with_columns(pl.lit(None).alias(col))
        aligned_dfs.append(df.select(sorted(all_columns)))
    
    return pl.concat(aligned_dfs).unique(subset=["iid"], keep="last")   

df = load_data()

if df.is_empty():
    st.error("No data loaded. Please check GCS connection or ensure data files exist.")
    st.stop()

st.multiselect(
    "Filter nach Labels",
    options=df.explode(LABELS_COLUMN)[LABELS_COLUMN].unique().drop_nulls().sort().to_list(),
    key="label_filter"
)

st.multiselect(
    "Filter nach Seite",
    options=sorted(df["form_page"].unique().drop_nulls().to_list()),
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

st.multiselect(
    "Filter nach Autor*in",
    options=sorted(df["author_name"].unique().drop_nulls().to_list()),
    key="author_filter"
)

st.multiselect(
    "Filter nach Organisation",
    options=sorted(df["org"].unique().drop_nulls().to_list()),
    key="org_filter"
)

st.multiselect(
    "Filter nach Feedback-Runde",
    options=sorted(df["feedback_round"].unique().drop_nulls().to_list()),
    key="feedback_round_filter"
)

st.divider()

st.subheader("Stichproben der Issues")

st.button("Neue Stichprobe ziehen")

df = df.filter(
    pl.all_horizontal([pl.col(LABELS_COLUMN).list.contains(label) for label in st.session_state.label_filter]) if st.session_state.label_filter else pl.lit(True)
).filter(
    pl.col("form_page").is_in(st.session_state.page_filter) if st.session_state.page_filter else pl.lit(True)
).filter(
    (pl.col("sentiment") >= st.session_state.sentiment_filter[0]) & 
    (pl.col("sentiment") <= st.session_state.sentiment_filter[1])
).filter(
    pl.col("author_name").is_in(st.session_state.author_filter) if "author_filter" in st.session_state and st.session_state.author_filter else pl.lit(True)
).filter(
    pl.col("desc_clean").str.to_lowercase().str.contains(st.session_state.search_term.lower()) if "search_term" in st.session_state and st.session_state.search_term else pl.lit(True)
).filter(
    pl.col("org").is_in(st.session_state.org_filter) if "org_filter" in st.session_state and st.session_state.org_filter else pl.lit(True)
).filter(
    pl.col("feedback_round").is_in(st.session_state.feedback_round_filter) if "feedback_round_filter" in st.session_state and st.session_state.feedback_round_filter else pl.lit(True)
)

if df.height == 0:
    st.warning(
        "Keine Issues entsprechen der aktuellen Filterkombination. Bitte passe die Filter an."
    )
    st.stop()


sample = df.sample(1) if df.height > 0 else pl.DataFrame(schema=df.schema)
if sample.height > 0:
    row = sample.row(0, named=True)
    st.markdown(f"""
**Titel:** {row['title']}

**Inhalt:** {row['desc_clean']}

**Aus Seite:** {row['form_page']}

**Labels:** {', '.join(row[LABELS_COLUMN]) if row[LABELS_COLUMN] else 'None'}

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
    df.explode(LABELS_COLUMN).group_by(LABELS_COLUMN).agg(pl.len()).sort("len", descending=True).rename({LABELS_COLUMN: "Label", "len": "Anzahl Issues"})
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
    df.explode(LABELS_COLUMN).group_by(LABELS_COLUMN).agg(pl.col("sentiment").mean()).sort("sentiment", descending=True).rename({LABELS_COLUMN: "Label", "sentiment": "Durchschnittlicher Sentiment"})
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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Autor*innen der Issues")

    st.dataframe(
        df.group_by("author_name")
            .agg(pl.len().alias("Anzahl Issues"))
            .rename({"author_name": "Autor*in"})
            .sort("Anzahl Issues", descending=True)
    )

with col2:
    st.subheader("Erwähnte Organisation")

    st.dataframe(
        df.group_by("org")
        .agg(pl.len().alias("Anzahl Issues"))
        .rename({"org": "Organisation"})
        .sort("Anzahl Issues", descending=True)
    )

st.divider()

st.subheader("Stichwortsuche")

st.text("Du suchst nach einem Thema, das nicht über die Labels abgedeckt ist? ")

st.text_input(
    "Suchbegriff",
    key="search_term"
)

st.divider()

st.subheader("Contribute")

st.markdown("""
Dieses Projekt ist Open Source und auf [GitHub](https://github.com/JuliaSchoepp/D-Stack-Analysis) verfügbar.
Contributions sind willkommen! Falls du Verbesserungen, Bug-Fixes oder neue Features hast, öffne gerne einen Pull Request.
""")

st.divider()

st.subheader("Impressum")

st.markdown("""
**Impressum:**
https://d-64.org/impressum
""")