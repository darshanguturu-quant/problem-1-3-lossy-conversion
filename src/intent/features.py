"""
Lexical, Syntactic, and Temporal Feature Engineering for Task 2 User Intent Classification.
Extracts 34 dense grammatical and semantic cues alongside sparse word and character n-gram TF-IDF vectors.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def normalize_query(text: str) -> str:
    """Normalize query text using Unicode NFKC, lowercase, and whitespace stripping."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


class GrammaticalCueExtractor:
    """Extracts 34 handcrafted dense grammatical, lexical, temporal, and structural cues."""

    # 1. Temporal anchors & intervals (5 cues)
    RE_TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
    RE_REL_START = re.compile(r"\b(at the beginning|in the beginning|first few seconds|opening scene|start of)\b", re.I)
    RE_REL_END = re.compile(r"\b(at the end|in the end|finally|last scene|closing|after the end)\b", re.I)
    RE_INTERVAL = re.compile(r"\b(during|between|while|throughout|continuous|how long|duration)\b", re.I)
    RE_SEQUENCE = re.compile(r"\b(after|before|next|then|subsequent|earlier|later|previously)\b", re.I)

    # 2. Visual / OCR / Frame-level markers (6 cues)
    RE_OCR = re.compile(r"\b(text|written|word|character|letter|sign|scoreboard|banner|subtitle|caption|displayed|font|number)\b", re.I)
    RE_VISUAL_ATTR = re.compile(r"\b(color|colour|wearing|dressed|clothes|shirt|pants|jacket|hat|hair|shoes|glasses)\b", re.I)
    RE_LOGO_ENTITY = re.compile(r"\b(logo|brand|icon|symbol|flag|license plate|face|expression|badge|emblem)\b", re.I)
    RE_SPATIAL_INSTANT = re.compile(r"\b(where is|top right|top left|bottom right|bottom left|background|foreground|on the screen|beside)\b", re.I)
    RE_FRAME_NOUNS = re.compile(r"\b(frame|screenshot|still|snapshot|image|freeze|photo|picture|view|instant)\b", re.I)
    RE_SINGLE_OBJECT = re.compile(r"\b(which object|what object|which item|what item|what phone|what car|what tool|what animal)\b", re.I)

    # 3. Event / Action / Clip-level markers (6 cues)
    RE_ACTION_VERBS = re.compile(r"\b(action|doing|performing|moves?|jump\w*|run\w*|kick\w*|score\w*|swim\w*|catch\w*|throw\w*|shoot\w*|dance\w*|sing\w*|play\w*|laugh\w*|fall\w*)\b", re.I)
    RE_ACTION_METHOD = re.compile(r"\b(how did|how do|how does|manner|method|by doing|purpose of|why did|how to)\b", re.I)
    RE_EVENT_ANCHOR = re.compile(r"\b(when does|when did|the moment|at what point|what happens when|which event|timestamp when)\b", re.I)
    RE_PROCESS = re.compile(r"\b(process|steps|procedure|interaction|reaction|change into|turn into|evolve|mechanism)\b", re.I)
    RE_CLIP_REQUEST = re.compile(r"\b(show the|find the part|clip|segment|highlight|scene where|play the|part where)\b", re.I)
    RE_CAUSE_EFFECT = re.compile(r"\b(caused by|resulted in|lead to|consequence|because of|reason for)\b", re.I)

    # 4. Synopsis / Whole-video markers (6 cues)
    RE_SYNOPSIS_NOUNS = re.compile(r"\b(summary|synopsis|overview|gist|takeaway|theme|topic|subject|focus|premise|storyline)\b", re.I)
    RE_SYNOPSIS_VERBS = re.compile(r"\b(summarize|describe the video|depict|illustrate|explain the video|tell about|what does the video depict)\b", re.I)
    RE_HOLISTIC_PHRASE = re.compile(r"\b(entire video|whole video|throughout the video|all in all|in general|overall|primary focus)\b", re.I)
    RE_TOTAL_COUNT = re.compile(r"\b(how many total|total number|in total|how many times across|all together|throughout the whole)\b", re.I)
    RE_MAIN_QUESTION = re.compile(r"\b(what is the video about|what is the primary focus|main topic of this video|best title)\b", re.I)
    RE_CONCLUSION = re.compile(r"\b(main conclusion|moral|lesson|final outcome of the video|overall message)\b", re.I)

    # 5. Interrogative types & syntax (7 cues)
    RE_WH_WHAT = re.compile(r"\bwhat\b", re.I)
    RE_WH_WHEN = re.compile(r"\bwhen\b", re.I)
    RE_WH_WHERE = re.compile(r"\bwhere\b", re.I)
    RE_WH_WHO = re.compile(r"\bwho\b", re.I)
    RE_WH_WHY = re.compile(r"\bwhy\b", re.I)
    RE_WH_HOW = re.compile(r"\bhow\b", re.I)
    RE_IMPERATIVE = re.compile(r"^(show|find|locate|get|give|tell|explain|summarize|list|identify)\b", re.I)

    NUM_FEATURES = 34

    @classmethod
    def extract_cues(cls, text: str) -> np.ndarray:
        """Extract a 34-dimensional dense feature vector from normalized query text."""
        t = normalize_query(text)
        features = np.zeros(cls.NUM_FEATURES, dtype=np.float32)

        # 1. Temporal cues
        features[0] = 1.0 if cls.RE_TIMESTAMP.search(t) else 0.0
        features[1] = 1.0 if cls.RE_REL_START.search(t) else 0.0
        features[2] = 1.0 if cls.RE_REL_END.search(t) else 0.0
        features[3] = 1.0 if cls.RE_INTERVAL.search(t) else 0.0
        features[4] = 1.0 if cls.RE_SEQUENCE.search(t) else 0.0

        # 2. Visual / OCR / Frame cues
        features[5] = 1.0 if cls.RE_OCR.search(t) else 0.0
        features[6] = 1.0 if cls.RE_VISUAL_ATTR.search(t) else 0.0
        features[7] = 1.0 if cls.RE_LOGO_ENTITY.search(t) else 0.0
        features[8] = 1.0 if cls.RE_SPATIAL_INSTANT.search(t) else 0.0
        features[9] = 1.0 if cls.RE_FRAME_NOUNS.search(t) else 0.0
        features[10] = 1.0 if cls.RE_SINGLE_OBJECT.search(t) else 0.0

        # 3. Action / Event / Clip cues
        features[11] = 1.0 if cls.RE_ACTION_VERBS.search(t) else 0.0
        features[12] = 1.0 if cls.RE_ACTION_METHOD.search(t) else 0.0
        features[13] = 1.0 if cls.RE_EVENT_ANCHOR.search(t) else 0.0
        features[14] = 1.0 if cls.RE_PROCESS.search(t) else 0.0
        features[15] = 1.0 if cls.RE_CLIP_REQUEST.search(t) else 0.0
        features[16] = 1.0 if cls.RE_CAUSE_EFFECT.search(t) else 0.0

        # 4. Synopsis / Whole-video cues
        features[17] = 1.0 if cls.RE_SYNOPSIS_NOUNS.search(t) else 0.0
        features[18] = 1.0 if cls.RE_SYNOPSIS_VERBS.search(t) else 0.0
        features[19] = 1.0 if cls.RE_HOLISTIC_PHRASE.search(t) else 0.0
        features[20] = 1.0 if cls.RE_TOTAL_COUNT.search(t) else 0.0
        features[21] = 1.0 if cls.RE_MAIN_QUESTION.search(t) else 0.0
        features[22] = 1.0 if cls.RE_CONCLUSION.search(t) else 0.0

        # 5. WH-words and syntax
        features[23] = 1.0 if cls.RE_WH_WHAT.search(t) else 0.0
        features[24] = 1.0 if cls.RE_WH_WHEN.search(t) else 0.0
        features[25] = 1.0 if cls.RE_WH_WHERE.search(t) else 0.0
        features[26] = 1.0 if cls.RE_WH_WHO.search(t) else 0.0
        features[27] = 1.0 if cls.RE_WH_WHY.search(t) else 0.0
        features[28] = 1.0 if cls.RE_WH_HOW.search(t) else 0.0
        features[29] = 1.0 if cls.RE_IMPERATIVE.search(t) else 0.0

        # 6. Structural & length cues
        words = t.split()
        features[30] = min(len(t) / 100.0, 1.0)
        features[31] = min(len(words) / 25.0, 1.0)
        features[32] = 1.0 if "?" in text else 0.0
        features[33] = 1.0 if re.search(r"\d", t) else 0.0

        return features

    @classmethod
    def batch_extract(cls, texts: List[str]) -> np.ndarray:
        """Extract cues for a list of query strings, returning shape (N, 34)."""
        return np.vstack([cls.extract_cues(text) for text in texts])


class HybridFeaturePipeline:
    """Combines word/char n-gram sublinear TF-IDF with 34 dense grammatical cues."""

    def __init__(self, max_word_features: int = 5000, max_char_features: int = 5000):
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=max_word_features,
            preprocessor=normalize_query
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=max_char_features,
            preprocessor=normalize_query
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "HybridFeaturePipeline":
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("HybridFeaturePipeline is not fitted.")
        word_tfidf = self.word_vec.transform(texts).toarray()
        char_tfidf = self.char_vec.transform(texts).toarray()
        cues = GrammaticalCueExtractor.batch_extract(texts)
        return np.hstack([word_tfidf, char_tfidf, cues])

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)
