import csv
import html
import io
import os
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.predict import FashionClassifier

st.set_page_config(
    page_title="Atelier — Fashion, decoded.",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #29262d;
        --muted: #77727d;
        --accent: #9c4a59;
        --accent-dark: #713945;
        --paper: #f7f5f2;
        --line: #e9e4e0;
        --card: #fffdfa;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,245,242,.94); }
    [data-testid="stSidebar"] {
        background: #fffdfa;
        border-right: 1px solid var(--line);
    }
    .block-container { max-width: 1220px; padding-top: 1.65rem; padding-bottom: 4rem; }
    .atelier-nav {
        align-items: center; display: flex; justify-content: space-between;
        padding: 0 0 1.15rem; border-bottom: 1px solid var(--line); margin-bottom: 1.35rem;
    }
    .atelier-logo {
        align-items: center; color: var(--ink); display: flex; gap: .65rem;
        font-size: 1.12rem; font-weight: 800; letter-spacing: -.045em;
    }
    .atelier-logo-mark {
        align-items: center; background: var(--accent-dark); border-radius: 11px;
        color: #fff; display: inline-flex; height: 34px; justify-content: center;
        width: 34px; font-size: 1.15rem;
    }
    .atelier-nav-note {
        color: var(--muted); font-size: .79rem; letter-spacing: .025em;
    }
    .atelier-hero {
        background:
          radial-gradient(ellipse at 90% 4%, rgba(219,156,162,.23), transparent 34%),
          linear-gradient(118deg, #29242c 0%, #49323c 57%, #77434d 100%);
        border-radius: 25px; padding: 2.45rem 2.65rem; color: #fff;
        box-shadow: 0 18px 42px rgba(49,34,41,.16); margin: .4rem 0 1.5rem;
        position: relative; overflow: hidden;
    }
    .atelier-hero::after {
        content: "✳"; position: absolute; color: rgba(255,255,255,.065);
        font-size: 15rem; right: 3%; top: -5.3rem; line-height: 1;
        pointer-events: none;
    }
    .atelier-hero-content { max-width: 720px; position: relative; z-index: 1; }
    .atelier-kicker {
        color: #e9c5cb; font-size: .7rem; font-weight: 800;
        letter-spacing: .19em; text-transform: uppercase;
    }
    .atelier-hero h1 {
        color: #fff; font-size: clamp(2.15rem, 5vw, 3.8rem); letter-spacing: -.055em;
        line-height: 1.02; margin: .7rem 0 .85rem; font-weight: 780;
    }
    .atelier-hero p { color: #f0e7e9; font-size: 1.02rem; line-height: 1.65; margin: 0; max-width: 620px; }
    .atelier-chip {
        display: inline-block; border: 1px solid rgba(255,255,255,.27);
        border-radius: 999px; padding: .4rem .72rem; color: #fff;
        font-size: .74rem; margin: 1.25rem .4rem 0 0;
    }
    .section-eyebrow {
        color: var(--accent); font-size: .69rem; font-weight: 820;
        letter-spacing: .15em; text-transform: uppercase; margin: .3rem 0 .35rem;
    }
    .section-title {
        color: var(--ink); font-size: 1.48rem; font-weight: 760;
        letter-spacing: -.035em; margin: 0 0 .28rem;
    }
    .section-copy { color: var(--muted); font-size: .91rem; line-height: 1.55; margin: 0 0 1.15rem; }
    [data-testid="stTabs"] [role="tablist"] {
        background: #eeebe7; border-radius: 12px; gap: .25rem;
        padding: .27rem; margin: .5rem 0 1.4rem;
    }
    [data-testid="stTabs"] button[role="tab"] {
        border-radius: 9px; color: #625e66; font-size: .88rem; font-weight: 650;
        min-height: 2.6rem; padding-left: 1.05rem; padding-right: 1.05rem;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #fffdfa; color: var(--accent-dark);
        box-shadow: 0 2px 6px rgba(35,28,33,.08);
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
    [data-testid="stFileUploader"] {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 17px; padding: .55rem;
    }
    [data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 15px; padding: .8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .result-card {
        background: var(--card); border: 1px solid var(--line); border-radius: 18px;
        padding: 1.1rem 1.2rem; margin: .45rem 0;
    }
    .result-label {
        color: var(--muted); font-size: .67rem; font-weight: 800;
        letter-spacing: .13em; text-transform: uppercase;
    }
    .result-category {
        color: var(--ink); font-size: 1.65rem; font-weight: 780;
        letter-spacing: -.04em; margin: .18rem 0 .6rem; overflow-wrap: anywhere;
    }
    .confidence-pill {
        display: inline-block; background: #f4e7e8; color: var(--accent-dark);
        border-radius: 999px; padding: .37rem .67rem; font-size: .76rem; font-weight: 760;
    }
    .prediction-row { margin: .82rem 0; }
    .prediction-meta {
        display: flex; justify-content: space-between; gap: .75rem;
        color: #514d55; font-size: .82rem; margin-bottom: .32rem;
    }
    .prediction-track { height: 7px; background: #eee9e6; border-radius: 999px; overflow: hidden; }
    .prediction-fill {
        height: 100%; background: linear-gradient(90deg, #bd7580, #87424f);
        border-radius: inherit;
    }
    .sample-label {
        color: var(--accent-dark); font-size: .66rem; font-weight: 820;
        letter-spacing: .12em; text-transform: uppercase;
    }
    .empty-state {
        background: var(--card); border: 1px dashed #d8c8c6; border-radius: 19px;
        padding: 2.2rem 1.4rem; text-align: center; margin: .6rem 0 1rem;
    }
    .empty-icon { color: var(--accent); font-size: 2rem; margin-bottom: .35rem; }
    .empty-title { color: var(--ink); font-weight: 740; font-size: 1.03rem; }
    .empty-copy { color: var(--muted); font-size: .86rem; line-height: 1.5; margin-top: .3rem; }
    .how-card {
        height: 100%; background: var(--card); border: 1px solid var(--line);
        border-radius: 16px; padding: 1rem 1.1rem;
    }
    .how-number { color: var(--accent); font-size: .68rem; font-weight: 850; letter-spacing: .12em; }
    .how-title { color: var(--ink); font-weight: 740; margin: .35rem 0 .28rem; }
    .how-copy { color: var(--muted); font-size: .82rem; line-height: 1.5; }
    .footer {
        border-top: 1px solid var(--line); color: var(--muted); display: flex;
        justify-content: space-between; gap: .8rem; margin-top: 2.7rem;
        padding-top: 1.15rem; font-size: .76rem;
    }
    .stButton > button { border-radius: 10px; font-weight: 650; }
    div[data-testid="stProgressBar"] > div > div { background-color: var(--accent); }
    @media (max-width: 700px) {
        .block-container { padding: 1.1rem 1rem 3rem; }
        .atelier-nav-note { display: none; }
        .atelier-hero { padding: 1.6rem; border-radius: 19px; }
        .atelier-hero::after { right: -3rem; top: -3rem; }
        .footer { flex-direction: column; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model_path = Path(os.getenv("MODEL_PATH", "models/best_model.pth"))
max_upload_bytes = 10 * 1024 * 1024


@st.cache_resource
def load_classifier(path: str, modified_ns: int) -> FashionClassifier:
    return FashionClassifier(path)


def read_image(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        return source.convert("RGB")


def prediction_card(result: dict[str, Any], low_confidence: float) -> None:
    category = html.escape(str(result["category"]))
    confidence = float(result["confidence"])
    match_status = "Worth a second look" if confidence < low_confidence else "Top pick"
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">The closest match</div>
          <div class="result-category">{category}</div>
          <span class="confidence-pill">{confidence:.1%} confidence · {match_status}</span>
        </div>
        <div class="result-label" style="margin:1rem 0 .25rem">Other likely categories</div>
        """,
        unsafe_allow_html=True,
    )
    for prediction in result["top_k"]:
        label = html.escape(str(prediction["label"]))
        score = max(0.0, min(1.0, float(prediction["score"])))
        st.markdown(
            f"""
            <div class="prediction-row">
              <div class="prediction-meta"><span>{label}</span><strong>{score:.1%}</strong></div>
              <div class="prediction-track"><div class="prediction-fill" style="width:{score:.1%}"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_results(
    results: list[tuple[str, Image.Image, dict[str, Any]]],
    top_k: int,
    low_confidence: float,
    show_images: bool,
    compare_scores: bool,
) -> None:
    if not results:
        return

    st.markdown(
        f'<div class="section-eyebrow">Your results</div>'
        f'<div class="section-title">The style report <span style="color:#9c4a59">· {len(results)}{" piece" if len(results) == 1 else " pieces"}</span></div>'
        '<div class="section-copy">A closer look at each item, ranked by how well it matches a category.</div>',
        unsafe_allow_html=True,
    )
    rows = []
    number_of_columns = min(len(results), 2)
    for row_start in range(0, len(results), number_of_columns):
        row_results = results[row_start : row_start + number_of_columns]
        columns = st.columns(number_of_columns, gap="large")
        for column, (filename, image, result) in zip(columns, row_results):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{html.escape(filename)}**")
                    if show_images:
                        st.image(image, width="stretch")
                    prediction_card(result, low_confidence)
                    rows.append(
                        {
                            "Image": filename,
                            "Best match": result["category"],
                            "Confidence": f"{float(result['confidence']):.1%}",
                            **{
                                f"Alternative {index}": f"{item['label']} ({float(item['score']):.1%})"
                                for index, item in enumerate(result["top_k"][1:], start=1)
                            },
                        }
                    )

    if compare_scores and len(results) > 1:
        st.markdown("#### Confidence across your edit")
        for name, _, result in results:
            confidence = max(0.0, min(1.0, float(result["confidence"])))
            st.markdown(
                f"""
                <div class="prediction-row">
                  <div class="prediction-meta">
                    <span>{html.escape(name)}</span><strong>{confidence:.1%}</strong>
                  </div>
                  <div class="prediction-track">
                    <div class="prediction-fill" style="width:{confidence:.1%}"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    st.download_button(
        "↓  Download style report",
        data=csv_buffer.getvalue(),
        file_name="atelier-style-report.csv",
        mime="text/csv",
        key=f"download-report-{len(rows)}",
    )


with st.sidebar:
    st.markdown("### ✳ Your studio settings")
    st.caption("Fine-tune the results to fit the way you like to explore.")
    top_k = st.slider("Categories to show", min_value=1, max_value=5, value=3)
    low_confidence = st.slider(
        "Flag below this confidence",
        min_value=0,
        max_value=100,
        value=35,
        step=5,
        format="%d%%",
    ) / 100
    show_images = st.toggle("Show image previews", value=True)
    compare_scores = st.toggle("Compare confidence scores", value=True)
    st.divider()
    st.markdown("**Image guide**")
    st.caption("JPG · PNG · WEBP")
    st.caption("Up to 4 photos · 10 MB per image")

st.markdown(
    """
    <header class="atelier-nav">
      <div class="atelier-logo"><span class="atelier-logo-mark">✳</span> atelier</div>
      <div class="atelier-nav-note">A little more insight for your next great find.</div>
    </header>
    <section class="atelier-hero">
      <div class="atelier-hero-content">
        <div class="atelier-kicker">Your AI-powered styling studio</div>
        <h1>Fashion, decoded.</h1>
        <p>Discover what makes a piece your own. Explore our style library or bring a few favourites for an instant, side-by-side fashion read.</p>
        <span class="atelier-chip">✦ &nbsp; Image recognition</span>
        <span class="atelier-chip">◉ &nbsp; Ranked predictions</span>
        <span class="atelier-chip">⌁ &nbsp; A smarter style edit</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if not model_path.is_file():
    st.error(f"Model checkpoint not found: {model_path}. Train a model first.")
else:
    try:
        classifier = load_classifier(str(model_path), model_path.stat().st_mtime_ns)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        st.error(f"Unable to load the classifier: {error}")
    else:
        accuracy = getattr(classifier, "val_accuracy", None)
        accuracy_label = f"{accuracy:.0%}" if accuracy is not None else "Ready"
        metric_columns = st.columns(3)
        metric_columns[0].metric("Style categories", len(classifier.class_names), "trained labels")
        metric_columns[1].metric(
            "AI model", classifier.model_name.replace("_", " ").title(), "image classifier"
        )
        metric_columns[2].metric("Validation score", accuracy_label, "best checkpoint")

        sample_root = Path(os.getenv("DATA_ROOT", "data")) / "test"
        sample_by_class = {
            folder.name: sorted(
                path for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            for folder in sample_root.iterdir()
            if folder.is_dir()
        } if sample_root.is_dir() else {}
        preferred_samples = ["dress", "t-shirt", "shoes", "outwear"]
        sample_classes = [
            label for label in preferred_samples
            if label in classifier.class_names and sample_by_class.get(label)
        ]
        if len(sample_classes) < 4:
            sample_classes.extend(
                label for label in classifier.class_names
                if label not in sample_classes and sample_by_class.get(label)
            )
        sample_classes = sample_classes[:4]

        sample_tab, upload_tab = st.tabs(["✳  Explore the style library", "＋  Upload your own"])
        with sample_tab:
            st.markdown(
                '<div class="section-eyebrow">01 / The style library</div>'
                '<div class="section-title">A little inspiration</div>'
                '<div class="section-copy">Pick a sample to see the classifier in action. These demo images are from the locally downloaded test set.</div>',
                unsafe_allow_html=True,
            )
            if not sample_classes:
                st.markdown(
                    """
                    <div class="empty-state">
                      <div class="empty-icon">✧</div>
                      <div class="empty-title">Bring your own inspiration</div>
                      <div class="empty-copy">The demo gallery isn't installed in this workspace. Open the upload tab to try your own product photo.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                sample_columns = st.columns(len(sample_classes), gap="medium")
                for column, label in zip(sample_columns, sample_classes):
                    sample_path = sample_by_class[label][0]
                    with column:
                        with st.container(border=True):
                            st.image(sample_path, width="stretch")
                            st.markdown(
                                f'<div class="sample-label">{html.escape(label.replace("-", " ").title())}</div>',
                                unsafe_allow_html=True,
                            )
                            st.caption("Explore this look")
                            if st.button(
                                "Discover this piece  →",
                                key=f"sample-{label}",
                                type="primary" if label == sample_classes[0] else "secondary",
                                width="stretch",
                            ):
                                try:
                                    sample_image = read_image(sample_path.read_bytes())
                                    with st.spinner("Reading the details…"):
                                        sample_result = classifier.predict(sample_image, top_k=top_k)
                                    st.session_state["atelier_sample_result"] = (
                                        str(sample_path),
                                        sample_result,
                                        top_k,
                                    )
                                except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                                    st.error(f"Unable to open the demo image: {error}")

            saved_sample = st.session_state.get("atelier_sample_result")
            if saved_sample and Path(saved_sample[0]).is_file():
                st.markdown("---")
                sample_path = Path(saved_sample[0])
                sample_image = read_image(sample_path.read_bytes())
                sample_result = saved_sample[1]
                if len(saved_sample) < 3 or saved_sample[2] != top_k:
                    sample_result = classifier.predict(sample_image, top_k=top_k)
                    st.session_state["atelier_sample_result"] = (
                        str(sample_path),
                        sample_result,
                        top_k,
                    )
                render_results(
                    [(sample_path.name, sample_image, sample_result)],
                    top_k,
                    low_confidence,
                    show_images,
                    compare_scores,
                )

        with upload_tab:
            st.markdown(
                '<div class="section-eyebrow">02 / Your personal edit</div>'
                '<div class="section-title">Bring your own pieces</div>'
                '<div class="section-copy">Upload one or up to four clear product photos to compare their closest categories and confidence scores.</div>',
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "Choose product photos",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                max_upload_size=10,
                help="JPG, PNG, or WebP. Up to four images; 10 MB each.",
                key="atelier_uploads",
            )
            if len(uploaded_files) > 4:
                st.warning("Choose up to four photos for a clear side-by-side comparison.")
            elif not uploaded_files:
                st.markdown(
                    """
                    <div class="empty-state">
                      <div class="empty-icon">↟</div>
                      <div class="empty-title">Make room for a new favourite</div>
                      <div class="empty-copy">Choose your images above. Your style report will appear right here.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                results: list[tuple[str, Image.Image, dict[str, Any]]] = []
                with st.spinner("Looking closely at your pieces…"):
                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.getvalue()
                        if len(file_bytes) > max_upload_bytes:
                            st.warning(f"{uploaded_file.name} is larger than 10 MB. Please choose a smaller image.")
                            continue
                        try:
                            image = read_image(file_bytes)
                            result = classifier.predict(image, top_k=top_k)
                        except (
                            OSError,
                            ValueError,
                            UnidentifiedImageError,
                            Image.DecompressionBombError,
                        ) as error:
                            st.warning(f"Couldn't read {uploaded_file.name}: {error}")
                        else:
                            results.append((uploaded_file.name, image, result))
                render_results(results, top_k, low_confidence, show_images, compare_scores)

        with st.expander("How Atelier reads a look"):
            how_columns = st.columns(3, gap="medium")
            steps = [
                ("01", "Choose a piece", "Try a gallery sample or upload a clear photo of your fashion find."),
                ("02", "Meet your matches", "The model ranks the closest styles and shows how confident it feels."),
                ("03", "Make your edit", "Compare a few pieces, tune the view, or download a report to keep."),
            ]
            for column, (number, title, copy) in zip(how_columns, steps):
                with column:
                    st.markdown(
                        f'<div class="how-card"><div class="how-number">{number} / THE EDIT</div>'
                        f'<div class="how-title">{title}</div><div class="how-copy">{copy}</div></div>',
                        unsafe_allow_html=True,
                    )

st.markdown(
    '<footer class="footer"><span>✳ ATELIER — Fashion, decoded.</span>'
    '<span>Image intelligence for inspiration. Predictions are suggestions, not product guarantees.</span></footer>',
    unsafe_allow_html=True,
)
