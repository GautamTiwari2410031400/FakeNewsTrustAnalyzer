import streamlit as st
import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from analyzer import analyze_article, analyze_article_url
from database import save_analysis, get_analysis_history


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="News Trust Analyzer",
    page_icon="📰",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .app-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 1.8rem;
        margin-bottom: 0.8rem;
    }

    .result-card {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        min-height: 120px;
    }

    .result-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.4rem;
    }

    .result-value {
        font-size: 1.65rem;
        font-weight: 700;
    }

    .claim-card {
        border-left: 4px solid #888;
        padding: 0.9rem 1rem;
        margin: 0.7rem 0;
        background: rgba(128, 128, 128, 0.05);
        border-radius: 5px;
    }

    .history-item {
        padding: 0.5rem 0;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">News Trust Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'NLP-based credibility assessment with explainable AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ABOUT
# ============================================================

with st.expander("About the system"):

    st.write(
        "This research prototype evaluates news articles using "
        "machine learning and multiple supporting credibility signals."
    )

    st.markdown("""
    **Machine Learning**  
    Linear SVM with TF-IDF features predicts whether an article is likely Fake or Real.

    **Linguistic Analysis**  
    Examines punctuation, questions, exclamation marks and uppercase words.

    **Clickbait Detection**  
    Identifies sensational headline patterns.

    **Contradiction Analysis**  
    Uses Natural Language Inference to identify strong contradictions between statements.

    **Claim Analysis**  
    Extracts potentially important claims that may require verification.

    **Explainable AI**  
    SHAP and LIME identify text features influencing the ML prediction.

    **Trust Score**  
    Combines the different signals into a score from 0 to 100.
    """)

    st.caption(
        "The Trust Score is an analytical estimate and should not "
        "be treated as proof that an article is true or false."
    )


# ============================================================
# ARTICLE INPUT
# ============================================================

st.markdown(
    '<div class="section-title">Article Input</div>',
    unsafe_allow_html=True
)

title = st.text_input(
    "Article title",
    placeholder="Enter the news headline"
)

url = st.text_input(
    "Article URL (optional)",
    placeholder="https://example.com/article"
)

article_text = st.text_area(
    "Article text",
    height=280,
    placeholder="Paste the article text here..."
)

analyze_button = st.button(
    "Analyze Article",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not article_text.strip() and not url.strip():

        st.warning(
            "Please enter either article text or an article URL."
        )

    else:

        with st.spinner("Analyzing article..."):

            # URL-based analysis
            if url.strip() and not article_text.strip():

                result = analyze_article_url(url.strip())

                if not result["success"]:
                    st.error(result["error"])
                    st.stop()

            # Direct text analysis
            else:

                result = analyze_article(
                    title=title,
                    article_text=article_text,
                    url=url
                )

        save_analysis(
            result=result,
            title=title,
            url=url
        )

        st.success("Analysis completed.")


        # ====================================================
        # ASSESSMENT
        # ====================================================

        st.markdown(
            '<div class="section-title">Assessment</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Prediction</div>
                    <div class="result-value">
                        {result["prediction"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Trust Score</div>
                    <div class="result-value">
                        {result["trust_score"]}
                    </div>
                    <div class="result-label">out of 100</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Trust Level</div>
                    <div class="result-value">
                        {result["trust_level"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Risk Level</div>
                    <div class="result-value">
                        {result["risk_level"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # SIGNAL ANALYSIS
        # ====================================================

        st.markdown(
            '<div class="section-title">Signal Analysis</div>',
            unsafe_allow_html=True
        )

        s1, s2, s3, s4, s5 = st.columns(5)

        s1.metric(
            "ML Score",
            result["ml_score"]
        )

        s2.metric(
            "Clickbait",
            result["clickbait_score"]
        )

        s3.metric(
            "Style Risk",
            result["style_risk"]
        )

        s4.metric(
            "Contradiction",
            result["contradiction_risk"]
        )

        s5.metric(
            "Source",
            result["source_score"]
        )


        # ====================================================
        # SHAP
        # ====================================================

        with st.expander("SHAP Explanation"):

            shap_explanations = result.get(
                "shap_explanations",
                []
            )

            if shap_explanations:

                for item in shap_explanations:

                    st.write(
                        f"**{item['feature']}** — "
                        f"{item['direction']} "
                        f"({item['shap_value']:.4f})"
                    )

            else:

                st.info(
                    "SHAP explanation is not available."
                )


        # ====================================================
        # LIME
        # ====================================================

        with st.expander("LIME Explanation"):

            lime_explanations = result.get(
                "lime_explanations",
                []
            )

            if lime_explanations:

                for item in lime_explanations:

                    direction = (
                        "supports the Real prediction"
                        if item["weight"] > 0
                        else "supports the Fake prediction"
                    )

                    st.write(
                        f"**{item['feature']}** — "
                        f"{direction} "
                        f"({item['weight']:.4f})"
                    )

            else:

                st.info(
                    "LIME explanation is not available."
                )


        # ====================================================
        # CLAIMS
        # ====================================================

        with st.expander("Claims & Evidence Analysis"):

            claims = result.get(
                "claims",
                []
            )

            if claims:

                for claim in claims:

                    st.markdown(
                        f"""
                        <div class="claim-card">
                            <strong>
                                Claim {claim["claim_id"]}
                            </strong>
                            <br><br>
                            {claim["claim"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(
                        f"Status: **{claim['status']}**"
                    )

                    if claim["evidence"]:

                        st.write(
                            f"Evidence: {claim['evidence']}"
                        )

                    else:

                        st.write(
                            "Evidence: **Needs verification**"
                        )

            else:

                st.info(
                    "No claims were extracted."
                )


        # ====================================================
        # CONTRADICTION ANALYSIS
        # ====================================================

        with st.expander("Contradiction Analysis"):

            contradictions = result.get(
                "contradictions",
                []
            )

            if contradictions:

                for i, contradiction in enumerate(
                    contradictions,
                    start=1
                ):

                    st.write(
                        f"**Contradiction {i}**"
                    )

                    st.write(
                        f"**Statement A:** "
                        f"{contradiction['sentence_a']}"
                    )

                    st.write(
                        f"**Statement B:** "
                        f"{contradiction['sentence_b']}"
                    )

                    st.write(
                        f"**NLI Confidence:** "
                        f"{contradiction['confidence']:.4f}"
                    )

            else:

                st.success(
                    "No strong contradictions detected."
                )


# ============================================================
# ANALYSIS HISTORY
# ============================================================

with st.expander("Analysis History"):

    history = get_analysis_history()

    if history:

        for row in history:

            timestamp = row[1]
            history_title = row[2]
            history_url = row[3]
            prediction = row[4]
            trust_score = row[5]
            trust_level = row[6]
            risk_level = row[7]

            st.markdown(
                f"**{timestamp} — "
                f"{history_title or 'Untitled Article'}**"
            )

            st.write(
                f"Prediction: {prediction}  |  "
                f"Trust Score: {trust_score}  |  "
                f"Trust Level: {trust_level}  |  "
                f"Risk Level: {risk_level}"
            )

            if history_url:
                st.write(
                    f"URL: {history_url}"
                )

            st.divider()

    else:

        st.info(
            "No previous analyses available."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Explainable Fake News Trust Analyzer · Research Prototype'
    '</div>',
    unsafe_allow_html=True
)