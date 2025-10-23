import re
from types import string
from collections import Counter

def _white_space_fix(text: str) -> str: return " ".join(text.split())
def _remove_articles(text: str) -> str: return re.sub(r"\b(a|an|the)\b", " ", text)
def _remove_punc(text: str) -> str: return text.translate(str.maketrans("", "", string.punctuation))

def normalize(text: str) -> str:
    return _white_space_fix(_remove_articles(_remove_punc(text.lower())))

def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))

def f1_score(pred: str, gold: str) -> float:
    p_tokens, g_tokens = normalize(pred).split(), normalize(gold).split()
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return float(p_tokens == g_tokens)
    common = sum((Counter(p_tokens) & Counter(g_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    return 2 * precision * recall / (precision + recall)