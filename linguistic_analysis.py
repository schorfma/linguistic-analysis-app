
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Text, Tuple, Union

from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError
from pydantic.types import PositiveInt
import srsly
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token, MorphAnalysis
import streamlit
from stqdm import stqdm


os.environ["TOKENIZERS_PARALLELISM"] = str(True).lower()

class FeatureAnalysis(BaseModel):
    name: Text
    distribution: Optional[Dict[Text, PositiveInt]] = None


class Section(BaseModel):
    title: Text
    text: Text
    doc: Optional[Doc] = None
    features: Optional[
        Dict[
            Text,
            Union[
                FeatureAnalysis,
                Dict[Text, FeatureAnalysis]
            ]
        ]
    ] = None

    class Config:
        arbitrary_types_allowed: bool = True


def analyze_morphological_features(doc: Doc, pos_tag: Text) -> Dict[Text, FeatureAnalysis]:
    verbs: List[Token] = [
        token
        for token in doc
        if token.pos_ == pos_tag
    ]

    features: Dict[Text, FeatureAnalysis] = dict()

    for verb in verbs:
        morph_analysis: MorphAnalysis = verb.morph
        for feature, value in morph_analysis.to_dict().items():
            if not feature in features:
                features[feature] = FeatureAnalysis(
                    name=feature,
                    distribution=dict()
                )

            feature: FeatureAnalysis = features.get(feature)

            distribution: Dict[
                Text, PositiveInt
            ] = feature.distribution

            if value not in distribution:
                distribution[value] = 0

            distribution[value] += 1

    return {
        feature_name: feature_analysis
        for feature_name, feature_analysis in features.items()
        if feature_analysis.distribution
    }

TITLE = "Linguistic Analysis"
ICON = "📊"

streamlit.set_page_config(
    page_title=TITLE,
    page_icon=ICON
)

streamlit.title(ICON + " " + TITLE)

streamlit.subheader("Analyzers and Features")

ANALYZERS_COLUMN, POS_TAGS_COLUMN = streamlit.columns(2)

ANALYZERS: Dict[Text, Callable[[Doc, Text], Dict[Text, FeatureAnalysis]]] = {
    "Morphological Features": analyze_morphological_features
}

with ANALYZERS_COLUMN:
    SELECTED_ANALYZER_NAMES: List[Text] = streamlit.multiselect(
        "Select Analyzers",
        options=list(
            ANALYZERS.keys()
        )
    )

SELECTED_ANALYZERS: Dict[Text, Callable[[Doc, Text], Dict[Text, FeatureAnalysis]]] = {
    name: ANALYZERS[name]
    for name in SELECTED_ANALYZER_NAMES
}

POS_TAGS: List[Text] = [  # https://universaldependencies.org/u/pos/index.html
    ("ADJ", "Adjective"),
    ("ADV", "Adverb"),
    ("INTJ", "Interjection"),
    ("NOUN", "Noun"),
    ("PROPN", "Proper Noun"),
    ("VERB", "Verb")
]

with POS_TAGS_COLUMN:
    SELECTED_POS_TAGS: List[Tuple[Text, Text]] = streamlit.multiselect(
        "Select Part of Speech (PoS) Tags",
        options=POS_TAGS,
        format_func=lambda option: f'{option[1]} ("{option[0]}")'
    )

streamlit.subheader("Language and Language Model")

LANGUAGE_COLUMN, LANGUAGE_MODEL_COLUMN = streamlit.columns(2)

MODEL_NAMES_BY_LANGUAGE: Dict[Text, List[Text]] = {
    "English": [
        "en_core_web_md"#,
        #"en_core_web_trf"
    ]
}
with LANGUAGE_COLUMN:
    SELECTED_LANGUAGE: Text = streamlit.selectbox(
        "Select Language",
        options=list(
            MODEL_NAMES_BY_LANGUAGE.keys()
        )
    )

with LANGUAGE_MODEL_COLUMN:
    SELECTED_MODEL_NAME: Text = streamlit.selectbox(
        "Select Language Model",
        options=MODEL_NAMES_BY_LANGUAGE[SELECTED_LANGUAGE]
    )

with streamlit.spinner("Loading Language Model..."):
    SELECTED_MODEL: Language = spacy.load(SELECTED_MODEL_NAME)

streamlit.header("Sections to analyze")

if (
        "sections" not in streamlit.session_state
) or (
        streamlit.session_state.sections and
        streamlit.button(f"Clear {len(streamlit.session_state.sections)} Sections")
):
    SECTIONS: Dict[Text, Section] = dict()
    streamlit.session_state.sections = SECTIONS

USE_YAML_LIST_MODE: bool = streamlit.checkbox(
    "Use YAML List Mode (Advanced)"
)

if USE_YAML_LIST_MODE:
    YAML_SECTIONS_LIST: Text = streamlit.text_area(
        "Insert YAML list with keys 'title' and 'text'"
    )

    if YAML_SECTIONS_LIST:
        SECTION_LIST: List[Dict[Text, Text]] = srsly.yaml_loads(
            YAML_SECTIONS_LIST
        )

        for section_data in SECTION_LIST:
            streamlit.session_state.sections[section_data["title"]] = Section(
                **section_data
            )

else:
    with streamlit.form(
            key="section"
    ):
        streamlit.subheader("Add Section")
        SECTION_TITLE: Text = streamlit.text_input("Section Title")
        SECTION_TEXT: Text = streamlit.text_area("Section Text")
        SECTION_SUBMITTED: bool = streamlit.form_submit_button(
            "Add"
        )

        try:
            SECTION: Section = Section(
                title=SECTION_TITLE,
                text=SECTION_TEXT
            )
        except ValidationError:
            streamlit.warning("Invalid Section")
            SECTION = None

        if SECTION_SUBMITTED and SECTION:
            streamlit.session_state.sections[SECTION.title] = SECTION

for section_name, section in stqdm(
        streamlit.session_state.sections.items()
):
    streamlit.header(section.title)

    section.doc = SELECTED_MODEL(section.text)

    with streamlit.expander(
        f"Text of Section '{section.title}' "
        f"({len(section.doc)} Tokens)",
        expanded=False
    ):
        streamlit.text(section.text)

    if not section.features:
        section.features = dict()

    for analyzer_name, analyzer in SELECTED_ANALYZERS.items():
        section.features.update(
            {
                pos_tag: analyzer(section.doc, pos_tag)
                for pos_tag, pos_tag_description in SELECTED_POS_TAGS
            }
        )

    with streamlit.expander(
        f"Feature Analysis of Section '{section.title}'",
        expanded=False
    ):
        streamlit.write(
            {
                pos_tag: {
                    feature_name: feature_analysis.distribution
                    for feature_name, feature_analysis in pos_tag_features.items()
                }
                for pos_tag, pos_tag_features in section.features.items()
            }
        )

        streamlit.subheader("Relative Distribution")

        streamlit.write(
            {
                pos_tag: {
                    feature_name: {
                        key: (
                            count / sum(feature_analysis.distribution.values())
                        )
                        for key, count in feature_analysis.distribution.items()
                    }
                    for feature_name, feature_analysis in pos_tag_features.items()
                }
                for pos_tag, pos_tag_features in section.features.items()
            }
        )
