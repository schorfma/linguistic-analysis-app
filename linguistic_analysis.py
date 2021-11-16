
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Text, Tuple, Union

from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError
from pydantic.types import PositiveInt
import srsly
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token, MorphAnalysis
from spacy.glossary import GLOSSARY as TAGS_GLOSSARY
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


POS_TAGS: List[Tuple[Text, Text]] = [  # https://universaldependencies.org/u/pos/index.html
    ("ADJ", "Adjective"),
    ("ADV", "Adverb"),
    ("INTJ", "Interjection"),
    ("NOUN", "Noun"),
    ("PROPN", "Proper Noun"),
    ("VERB", "Verb")
]

POS_TAGS_DICT = dict(POS_TAGS)


def analyze_morphological_features(doc: Doc, pos_tag: Optional[Text] = None) -> Dict[Text, FeatureAnalysis]:
    filtered_words: List[Token] = [
        token
        for token in doc
        if (not pos_tag) or (token.pos_ == pos_tag)
    ]

    features: Dict[Text, FeatureAnalysis] = dict()

    for filtered_word in filtered_words:
        morph_analysis: MorphAnalysis = filtered_word.morph
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


def extract_word_families_by_lemma(doc: Doc) -> Dict[Text, Dict]:
    word_families: Dict[Text, Set[Token]] = dict()

    for token in doc:
        lemma: Text = token.lemma_

        if lemma and lemma.isalpha():
            if not lemma in word_families:
                word_families[lemma] = set()

            word_families[lemma].add(token)

    return {
        lemma: {
            f"{str(token.text).lower()} ({token.pos_})": {
                "Part of Speech (PoS) Tag": (
                    f"{token.pos_} ({TAGS_GLOSSARY[token.pos_]})"
                    if token.pos_ else None
                ),
                "Fine-grained Part of Speech (PoS) Tag": (
                    f"{token.tag_} ({TAGS_GLOSSARY[token.tag_]})"
                    if token.tag_ else None
                )
            }
            for token in word_family
        }
        for lemma, word_family in word_families.items()
        if word_family
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

ANALYZERS: Dict[Text, Callable[[Doc, Text], Dict[Text, Union[FeatureAnalysis, Dict]]]] = {
    "Morphological Features": analyze_morphological_features,
    "Word Families (by Lemma)": extract_word_families_by_lemma
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

with POS_TAGS_COLUMN:
    if "Morphological Features" in SELECTED_ANALYZERS:
        if streamlit.checkbox("Use all PoS Tags"):
            SELECTED_POS_TAGS = POS_TAGS
        else:
            SELECTED_POS_TAGS: List[Tuple[Text, Text]] = streamlit.multiselect(
                "Select Part of Speech (PoS) Tags",
                options=POS_TAGS,
                format_func=lambda option: f'{option[1]} ("{option[0]}")'
            )
    else:
        SELECTED_POS_TAGS = POS_TAGS

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
        if analyzer_name == "Morphological Features":
            section.features.update(
                {
                    pos_tag: analyzer(section.doc, pos_tag)
                    for pos_tag, pos_tag_description in SELECTED_POS_TAGS
                }
            )
        elif analyzer_name == "Word Families (by Lemma)":
            section.features["Word Families"] = analyzer(section.doc)

    with streamlit.expander(
        f"Feature Analysis of Section '{section.title}'",
        expanded=False
    ):
        if "Word Families (by Lemma)" in SELECTED_ANALYZERS:
            WORD_FAMILY_SIZE_THRESHOLD: PositiveInt = streamlit.number_input(
                "Only show word families of this size or higher",
                min_value=1,
                value=2
            )
        else:
            WORD_FAMILY_SIZE_THRESHOLD: PositiveInt = 1

        streamlit.write(
            {
                feature_group_key: (
                    {
                        feature_name: feature_analysis.distribution
                        for feature_name, feature_analysis in feature_group_features.items()
                    }
                    if not feature_group_key == "Word Families"
                    else {
                        word_family_lemma: word_family
                        for word_family_lemma, word_family in feature_group_features.items()
                        if len(word_family) >= WORD_FAMILY_SIZE_THRESHOLD
                    }
                )
                for feature_group_key, feature_group_features in section.features.items()
            }
        )

        if "Morphological Features" in SELECTED_ANALYZERS:

            streamlit.subheader("Relative Distribution")

            streamlit.write(
                {
                    feature_group: {
                        feature_name: {
                            key: (
                                round(
                                    count / sum(feature_analysis.distribution.values()),
                                    ndigits=2
                                )
                            )
                            for key, count in feature_analysis.distribution.items()
                        }
                        for feature_name, feature_analysis in feature_group_features.items()
                    }
                    for feature_group, feature_group_features in section.features.items()
                    if feature_group != "Word Families"
                }
            )
