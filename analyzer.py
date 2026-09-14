
import re
from pathlib import Path
import numpy as np
import joblib
import requests

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from itertools import combinations

import spacy
import shap
from transformers import pipeline

from lime.lime_text import LimeTextExplainer


# ============================================================
# LOAD SAVED MODELS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

svm_model = joblib.load(
    MODEL_DIR / "linear_svm_model.pkl"
)

tfidf_unique = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

explainer = joblib.load(
    MODEL_DIR / "shap_explainer.pkl"
)

nlp = spacy.load("en_core_web_sm")

nli_pipeline = pipeline(
    "text-classification",
    model="FacebookAI/roberta-large-mnli"
)

lime_explainer = LimeTextExplainer(
    class_names=["Fake", "Real"]
)


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text)
    text = BeautifulSoup(
        text,
        "html.parser"
    ).get_text(" ")

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# ============================================================
# ML ANALYSIS
# ============================================================

def get_prediction_label(prediction):
    if prediction == 1:
        return "Real"
    else:
        return "Fake"


def svm_trust_score(decision_score):
    score = 1 / (1 + np.exp(-decision_score))
    return round(float(score * 100), 2)


def analyze_article_ml(title, article_text):
    combined_text = (
        str(title) + " " + str(article_text)
    )

    features = tfidf_unique.transform(
        [combined_text]
    )

    prediction = svm_model.predict(
        features
    )[0]

    decision_score = svm_model.decision_function(
        features
    )[0]

    ml_score = svm_trust_score(
        decision_score
    )

    return {
        "prediction": get_prediction_label(
            prediction
        ),
        "ml_score": ml_score,
        "decision_score": round(
            float(decision_score),
            4
        )
    }


# ============================================================
# LINGUISTIC / STYLE ANALYSIS
# ============================================================

def extract_linguistic_features(text):
    text = str(text)

    features = {
        "char_count": len(text),
        "word_count": len(text.split()),
        "sentence_count": len(
            re.findall(r"[.!?]", text)
        ),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_count": len(
            re.findall(
                r"\b[A-Z]{2,}\b",
                text
            )
        )
    }

    return features


def calculate_style_risk(features):
    risk = 0

    if features["exclamation_count"] >= 2:
        risk += 35

    if features["question_count"] >= 2:
        risk += 30

    if features["uppercase_count"] >= 5:
        risk += 35

    return min(risk, 100)


def analyze_article_style(article_text):
    features = extract_linguistic_features(
        article_text
    )

    style_risk = calculate_style_risk(
        features
    )

    return {
        "style_features": features,
        "style_risk": style_risk
    }


# ============================================================
# CLICKBAIT ANALYSIS
# ============================================================

clickbait_phrases = [
    "you won't believe what",
    "you won't believe",
    "shocking truth",
    "what happens next",
    "this is what happened",
    "must see",
    "breaking",
    "urgent",
    "unbelievable",
    "mind blowing",
    "secret",
    "the truth about",
    "revealed",
    "shocking"
]


def detect_clickbait(title):
    title = str(title).lower()

    matched_phrases = []

    for phrase in clickbait_phrases:
        if phrase in title:
            if not any(
                phrase != other
                and phrase in other
                for other in matched_phrases
            ):
                matched_phrases.append(
                    phrase
                )

    exclamation_count = title.count("!")
    question_count = title.count("?")

    uppercase_words = re.findall(
        r"\b[A-Z]{2,}\b",
        str(title)
    )

    score = 0

    score += len(
        matched_phrases
    ) * 2

    if exclamation_count >= 2:
        score += 1

    if question_count >= 2:
        score += 1

    if len(uppercase_words) >= 2:
        score += 1

    clickbait_score = min(
        score * 20,
        100
    )

    return {
        "clickbait_score": clickbait_score,
        "matched_phrases": matched_phrases
    }


def analyze_article_clickbait(title):
    return detect_clickbait(title)


# ============================================================
# SENTENCE / CONTRADICTION ANALYSIS
# ============================================================

def extract_sentences(text):
    doc = nlp(str(text))

    sentences = [
        sentence.text.strip()
        for sentence in doc.sents
        if sentence.text.strip()
    ]

    return sentences


def prepare_contradiction_sentences(
    text,
    max_sentences=10
):
    return extract_sentences(
        text
    )[:max_sentences]


def check_contradiction(
    statement_a,
    statement_b
):
    result = nli_pipeline(
        f"{statement_a} </s></s> {statement_b}"
    )[0]

    return result


def calculate_contradiction_risk(
    contradictions
):
    if len(contradictions) == 0:
        return 0

    risk = len(
        contradictions
    ) * 50

    return min(risk, 100)


def detect_contradictions(
    text,
    max_sentences=10
):
    sentences = prepare_contradiction_sentences(
        text,
        max_sentences=max_sentences
    )

    contradictions = []

    for sentence_a, sentence_b in combinations(
        sentences,
        2
    ):
        result = check_contradiction(
            sentence_a,
            sentence_b
        )

        if result["label"] == "CONTRADICTION":
            contradictions.append({
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "confidence": result["score"]
            })

    return contradictions


def analyze_article_contradictions(
    article_text
):
    contradictions = detect_contradictions(
        article_text,
        max_sentences=10
    )

    contradiction_risk = calculate_contradiction_risk(
        contradictions
    )

    return {
        "contradictions": contradictions,
        "contradiction_risk": contradiction_risk
    }


# ============================================================
# CLAIM ANALYSIS
# ============================================================

def extract_claim_details(
    text,
    max_claims=5
):
    sentences = extract_sentences(
        text
    )

    claims = []

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) < 5:
            continue

        if sentence.endswith("?"):
            continue

        claims.append({
            "claim_id": len(claims) + 1,
            "claim": sentence,
            "word_count": len(
                sentence.split()
            )
        })

        if len(claims) >= max_claims:
            break

    return claims


def analyze_claims(claims):
    analyzed_claims = []

    for claim_data in claims:
        analyzed_claims.append({
            "claim_id": claim_data["claim_id"],
            "claim": claim_data["claim"],
            "status": "Needs verification",
            "evidence": None
        })

    return analyzed_claims


def analyze_article_claims(
    article_text
):
    claims = extract_claim_details(
        article_text,
        max_claims=5
    )

    analyzed_claims = analyze_claims(
        claims
    )

    return {
        "claims": analyzed_claims,
        "claim_count": len(
            analyzed_claims
        )
    }


# ============================================================
# SOURCE CREDIBILITY
# ============================================================

source_profiles = {
    "reuters.com": {
        "credibility_score": 90,
        "description":
            "Established international news organization"
    },
    "apnews.com": {
        "credibility_score": 90,
        "description":
            "Established international news organization"
    },
    "bbc.com": {
        "credibility_score": 85,
        "description":
            "Established international news organization"
    }
}


def extract_domain(url):
    parsed_url = urlparse(url)

    domain = parsed_url.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def analyze_source(url):
    domain = extract_domain(url)

    if domain in source_profiles:
        profile = source_profiles[domain]

        return {
            "domain": domain,
            "credibility_score":
                profile["credibility_score"],
            "description":
                profile["description"],
            "status": "Known source"
        }

    return {
        "domain": domain,
        "credibility_score": 50,
        "description":
            "Source not found in credibility database",
        "status": "Unknown source"
    }


def analyze_article_source(url):
    if not url:
        return {
            "domain": None,
            "credibility_score": 50,
            "description":
                "No source URL provided",
            "status": "No source"
        }

    return analyze_source(url)


# ============================================================
# SHAP EXPLANATION
# ============================================================

def get_shap_explanation(
    shap_values,
    feature_names,
    top_n=5
):
    shap_values = shap_values[0]

    top_indices = np.argsort(
        np.abs(shap_values)
    )[-top_n:][::-1]

    explanations = []

    for index in top_indices:
        feature = feature_names[index]
        value = shap_values[index]

        if value > 0:
            direction = (
                "supports the Real prediction"
            )
        else:
            direction = (
                "supports the Fake prediction"
            )

        explanations.append({
            "feature": feature,
            "shap_value": round(
                float(value),
                4
            ),
            "direction": direction
        })

    return explanations


def analyze_article_shap(
    title,
    article_text
):
    combined_text = (
        str(title) + " " + str(article_text)
    )

    features = tfidf_unique.transform(
        [combined_text]
    )

    shap_values = explainer.shap_values(
        features
    )

    feature_names = (
        tfidf_unique
        .get_feature_names_out()
    )

    return get_shap_explanation(
        shap_values,
        feature_names,
        top_n=5
    )


# ============================================================
# LIME EXPLANATION
# ============================================================

def predict_for_lime(texts):
    features = tfidf_unique.transform(
        texts
    )

    decision_scores = (
        svm_model.decision_function(
            features
        )
    )

    real_probability = (
        1 / (1 + np.exp(-decision_scores))
    )

    fake_probability = (
        1 - real_probability
    )

    return np.column_stack([
        fake_probability,
        real_probability
    ])


def analyze_article_lime(
    title,
    article_text
):
    combined_text = (
        str(title) + " " + str(article_text)
    )

    lime_result = (
        lime_explainer.explain_instance(
            combined_text,
            predict_for_lime,
            num_features=5,
            labels=[0, 1]
        )
    )

    return lime_result.as_list(
        label=1
    )


# ============================================================
# TRUST SCORE
# ============================================================

def calculate_combined_score(
    ml_score,
    clickbait_score,
    style_risk,
    contradiction_risk,
    source_score
):
    trust_score = (
        (ml_score * 0.45)
        + ((100 - clickbait_score) * 0.15)
        + ((100 - style_risk) * 0.10)
        + ((100 - contradiction_risk) * 0.15)
        + (source_score * 0.15)
    )

    return round(
        trust_score,
        2
    )


def get_trust_level(trust_score):
    if trust_score >= 75:
        return "High Trust"
    elif trust_score >= 50:
        return "Moderate Trust"
    elif trust_score >= 25:
        return "Low Trust"
    else:
        return "Very Low Trust"


def get_risk_level(trust_score):
    if trust_score >= 75:
        return "Low Risk"
    elif trust_score >= 50:
        return "Moderate Risk"
    elif trust_score >= 25:
        return "High Risk"
    else:
        return "Very High Risk"


# ============================================================
# URL EXTRACTION
# ============================================================

def extract_article_from_url(
    url,
    max_chars=20000
):
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for element in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header"
        ]):
            element.decompose()

        title = (
            soup.title.get_text(
                strip=True
            )
            if soup.title
            else ""
        )

        article_text = soup.get_text(
            separator=" ",
            strip=True
        )

        article_text = clean_text(
            article_text
        )

        if not article_text:
            raise ValueError(
                "No readable text found on this webpage."
            )

        article_text = article_text[
            :max_chars
        ]

        return {
            "success": True,
            "title": title,
            "article_text": article_text,
            "url": url,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "title": "",
            "article_text": "",
            "url": url,
            "error": str(e)
        }


def validate_article_text(
    article_text,
    min_chars=300
):
    text = str(
        article_text
    ).strip()

    if len(text) < min_chars:
        return {
            "valid": False,
            "message":
                "The extracted page does not contain enough text to analyze."
        }

    return {
        "valid": True,
        "message":
            "Article text is sufficient for analysis."
    }


# ============================================================
# MASTER ARTICLE ANALYZER
# ============================================================

def analyze_article(
    title,
    article_text,
    url=""
):
    ml_result = analyze_article_ml(
        title,
        article_text
    )

    style_result = analyze_article_style(
        article_text
    )

    clickbait_result = (
        analyze_article_clickbait(
            title
        )
    )

    contradiction_result = (
        analyze_article_contradictions(
            article_text
        )
    )

    claim_result = (
        analyze_article_claims(
            article_text
        )
    )

    source_result = (
        analyze_article_source(
            url
        )
    )

    shap_result = analyze_article_shap(
        title,
        article_text
    )

    lime_result = analyze_article_lime(
        title,
        article_text
    )

    trust_score = calculate_combined_score(
        ml_score=ml_result["ml_score"],
        clickbait_score=
            clickbait_result["clickbait_score"],
        style_risk=
            style_result["style_risk"],
        contradiction_risk=
            contradiction_result[
                "contradiction_risk"
            ],
        source_score=
            source_result[
                "credibility_score"
            ]
    )

    return {
        "prediction":
            ml_result["prediction"],
        "ml_score":
            ml_result["ml_score"],
        "clickbait_score":
            clickbait_result[
                "clickbait_score"
            ],
        "style_risk":
            style_result["style_risk"],
        "contradiction_risk":
            contradiction_result[
                "contradiction_risk"
            ],
        "source_score":
            source_result[
                "credibility_score"
            ],
        "trust_score":
            trust_score,
        "trust_level":
            get_trust_level(
                trust_score
            ),
        "risk_level":
            get_risk_level(
                trust_score
            ),
        "matched_clickbait_phrases":
            clickbait_result[
                "matched_phrases"
            ],
        "contradictions":
            contradiction_result[
                "contradictions"
            ],
        "claims":
            claim_result["claims"],
        "source":
            source_result,
        "shap_explanations":
            shap_result,
        "lime_explanations": [
            {
                "feature": str(feature),
                "weight": round(
                    float(weight),
                    4
                )
            }
            for feature, weight
            in lime_result
        ]
    }


# ============================================================
# URL ANALYZER
# ============================================================

def analyze_article_url(url):
    url_result = extract_article_from_url(
        url
    )

    if not url_result["success"]:
        return {
            "success": False,
            "error":
                url_result["error"]
        }

    validation = validate_article_text(
        url_result["article_text"]
    )

    if not validation["valid"]:
        return {
            "success": False,
            "error":
                validation["message"]
        }

    analysis = analyze_article(
        title=url_result["title"],
        article_text=
            url_result["article_text"],
        url=url
    )

    analysis["success"] = True
    analysis["title"] = (
        url_result["title"]
    )
    analysis["url"] = url

    return analysis
