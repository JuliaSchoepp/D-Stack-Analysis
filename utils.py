import requests
import polars as pl

def fetch_all_gitlab_issues(
    project_path: str,
    gitlab_base_url: str = "https://gitlab.opencode.de",
    per_page: int = 100,
    timeout: int = 30,
):
    """
    Fetch all issues from a public GitLab project.

    project_path: "namespace/project", e.g. "dstack/d-stack-home"
    returns: list of issue JSON objects
    """
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    # URL-encode project path
    project_path_enc = project_path.replace("/", "%2F")
    base_api = f"{gitlab_base_url}/api/v4/projects/{project_path_enc}/issues"

    issues = []
    page = 1

    while True:
        resp = session.get(
            base_api,
            params={
                "state": "all",
                "per_page": per_page,
                "page": page,
                "order_by": "created_at",
                "sort": "asc",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        issues.extend(batch)
        page += 1

    return issues

def clean_issues_df(df: pl.DataFrame, columns_to_use: list, rows_to_exclude: list) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("author").struct.field("id").alias("author_id"),
        pl.col("author").struct.field("name").alias("author_name"),
        pl.col("author").struct.field("state").alias("author_state"),
    ])
    df = df.with_columns([
        pl.col("created_at").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ").alias("created_at"),
        pl.col("updated_at").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ").alias("updated_at"),
        pl.col("closed_at").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ").alias("closed_at"),
    ])
    df = df.filter(~pl.col("iid").is_in(rows_to_exclude))
    df = df.unique(subset=["title", "description"])
    return df.select(columns_to_use)
    
def prepare_issues_df(df: pl.DataFrame, desc_to_exclude: list) -> pl.DataFrame:
    # clean description
    df = df.with_columns(
        desc_clean = pl.col("description").str.replace_all("**Feedback:** <br>", "", literal=True)
    )
    # get more insights on where issue comes from
    df = df.with_columns(
        is_from_form = pl.col("title").str.starts_with("Feedback für die Seite"),
        form_page = (
            pl.when(pl.col("title").str.starts_with("Feedback für die Seite"))
            .then(
                pl.col("title")
                .str.replace("^Feedback für die Seite", "")
                .str.strip_chars()
            )
            .otherwise(pl.lit("Via OpenCode"))
        )
    )
    df = df.filter(~pl.col("desc_clean").is_in(desc_to_exclude))
    return df

def postprocess_issues(df: pl.DataFrame) -> pl.DataFrame:
    """
    Postprocess the labeled issues DataFrame.
    - Add "Unklar" label where labels is empty
    - Exclude certain pages
    - Clean up form_page
    """
    # Add "Unklar" to empty labels
    df = df.with_columns(
        pl.when(pl.col("labels").list.len() == 0)
        .then(pl.lit(["Unklar"]))
        .otherwise(pl.col("labels"))
        .alias("labels")
    )
    
    # Exclude certain pages
    exclude_pages = ['/beteiligung?utm_source=chatgpt.com', '/wtf']
    df = df.filter(~pl.col("form_page").is_in(exclude_pages))
    
    # Clean form_page: remove trailing / unless it's "/", then replace with "home"
    df = df.with_columns(
        pl.when(pl.col("form_page") == "/")
        .then(pl.lit("home"))
        .otherwise(pl.col("form_page").str.strip_suffix("/"))
        .alias("form_page")
    )
    
    return df