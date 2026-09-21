"""Text corruption functions for CILN-Bench (text modality).

Four canonical corruptions, one per family, mirroring the NL-Augmenter taxonomy.
All deterministic given a numpy Generator. Severity controls perturbation
fraction with the same 1/3/5 grid we use for image/tabular:

    severity 1 -> 5%   of perturbable tokens
    severity 3 -> 15%
    severity 5 -> 30%

A "perturbable token" is a character (for Butter Fingers, Random Casing) or a
word (for Synonym, Word Swap).
"""
from __future__ import annotations
import random
import string
import numpy as np

SEVERITY_FRACTION = {1: 0.05, 3: 0.15, 5: 0.30, 6: 0.40, 8: 0.60}

# Adjacent-keyboard map for Butter Fingers.
_QWERTY = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy', 'y': 'tghu',
    'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv',
    'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
    'n': 'bhjm', 'm': 'njk',
}


def butter_fingers(text: str, severity: int, rng: np.random.Generator) -> tuple[str, dict]:
    """Replace ~severity% of alphabetic characters with an adjacent QWERTY key.
    Preserves case. Skips non-alphabetic characters."""
    frac = SEVERITY_FRACTION[severity]
    chars = list(text)
    alpha_idx = [i for i, c in enumerate(chars) if c.isalpha()]
    n_perturb = max(1, int(round(len(alpha_idx) * frac))) if alpha_idx else 0
    if n_perturb == 0:
        return text, {'n_perturbed': 0}
    pick = rng.choice(len(alpha_idx), size=n_perturb, replace=False)
    actually = 0
    for k in pick:
        i = alpha_idx[k]
        c = chars[i]
        lower = c.lower()
        if lower not in _QWERTY:
            continue
        neighbors = _QWERTY[lower]
        new = neighbors[int(rng.integers(0, len(neighbors)))]
        chars[i] = new.upper() if c.isupper() else new
        actually += 1
    return ''.join(chars), {'n_perturbed': actually, 'severity_fraction': frac}


def random_casing(text: str, severity: int, rng: np.random.Generator) -> tuple[str, dict]:
    """Randomly flip the case of ~severity% of alphabetic characters."""
    frac = SEVERITY_FRACTION[severity]
    chars = list(text)
    alpha_idx = [i for i, c in enumerate(chars) if c.isalpha()]
    n_perturb = max(1, int(round(len(alpha_idx) * frac))) if alpha_idx else 0
    if n_perturb == 0:
        return text, {'n_perturbed': 0}
    pick = rng.choice(len(alpha_idx), size=n_perturb, replace=False)
    for k in pick:
        i = alpha_idx[k]
        c = chars[i]
        chars[i] = c.lower() if c.isupper() else c.upper()
    return ''.join(chars), {'n_perturbed': len(pick), 'severity_fraction': frac}


def word_swap(text: str, severity: int, rng: np.random.Generator) -> tuple[str, dict]:
    """Swap ~severity% of adjacent word pairs."""
    frac = SEVERITY_FRACTION[severity]
    words = text.split()
    n_swappable = max(0, len(words) - 1)
    n_swap = max(1, int(round(n_swappable * frac))) if n_swappable else 0
    if n_swap == 0:
        return text, {'n_perturbed': 0}
    # Pick non-overlapping swap positions
    candidates = list(range(n_swappable))
    rng.shuffle(candidates)
    swapped_positions = []
    used = set()
    for pos in candidates:
        if len(swapped_positions) >= n_swap:
            break
        if pos in used or (pos + 1) in used:
            continue
        words[pos], words[pos + 1] = words[pos + 1], words[pos]
        swapped_positions.append(pos)
        used.add(pos); used.add(pos + 1)
    return ' '.join(words), {'n_perturbed': len(swapped_positions),
                              'severity_fraction': frac}


def synonym_substitution(text: str, severity: int, rng: np.random.Generator,
                          synonym_dict=None) -> tuple[str, dict]:
    """Replace ~severity% of words with a WordNet synonym (if one exists).
    The synonym_dict is loaded once and passed in to avoid NLTK overhead per call."""
    if synonym_dict is None:
        raise ValueError("synonym_dict must be provided (use load_wordnet() once)")
    frac = SEVERITY_FRACTION[severity]
    words = text.split()
    # Identify words that have at least one synonym in our dictionary
    candidates = [i for i, w in enumerate(words)
                   if w.lower().strip(string.punctuation) in synonym_dict]
    n_perturb = max(1, int(round(len(candidates) * frac))) if candidates else 0
    if n_perturb == 0:
        return text, {'n_perturbed': 0}
    pick = rng.choice(len(candidates), size=min(n_perturb, len(candidates)), replace=False)
    actually = 0
    for k in pick:
        i = candidates[k]
        w = words[i]
        bare = w.lower().strip(string.punctuation)
        synonyms = synonym_dict.get(bare, [])
        if not synonyms:
            continue
        # Pick a random synonym (deterministic given rng)
        new_bare = synonyms[int(rng.integers(0, len(synonyms)))]
        # Preserve case of first letter + trailing punctuation
        prefix_punct = ''.join(c for c in w[:len(w) - len(w.lstrip(string.punctuation))])
        suffix_punct = ''.join(c for c in w[len(w.rstrip(string.punctuation)):])
        if w[0].isupper():
            new_bare = new_bare.capitalize()
        words[i] = prefix_punct + new_bare + suffix_punct
        actually += 1
    return ' '.join(words), {'n_perturbed': actually, 'severity_fraction': frac}


def load_wordnet_synonyms():
    """Build a {word -> [synonyms]} dictionary from WordNet. Called once at startup."""
    try:
        from nltk.corpus import wordnet
        # NLTK needs the wordnet data; try once.
        try:
            wordnet.synsets('test')
        except LookupError:
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    except ImportError:
        raise ImportError("nltk is required for synonym substitution. pip install nltk")
    from nltk.corpus import wordnet

    # We only build the dict on a query-time basis to keep startup fast.
    # Use a cache.
    cache: dict[str, list[str]] = {}

    def get_synonyms(word: str) -> list[str]:
        if word in cache:
            return cache[word]
        syns = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name != word and ' ' not in name:
                    syns.add(name)
        result = sorted(syns)
        cache[word] = result
        return result

    # Returns a dict-like proxy
    class _SynonymProxy:
        def __contains__(self, w): return bool(get_synonyms(w))
        def get(self, w, default=None): return get_synonyms(w) or (default or [])

    return _SynonymProxy()


def word_deletion(text: str, severity: int, rng: np.random.Generator) -> tuple[str, dict]:
    """Delete ~severity% of words entirely (information loss, not reordering)."""
    frac = SEVERITY_FRACTION[severity]
    words = text.split()
    n_del = max(1, int(round(len(words) * frac))) if words else 0
    if n_del == 0 or n_del >= len(words):
        return text, {'n_perturbed': 0}
    keep_n = len(words) - n_del
    keep_idx = sorted(rng.choice(len(words), size=keep_n, replace=False))
    out_words = [words[i] for i in keep_idx]
    return ' '.join(out_words), {'n_perturbed': n_del, 'severity_fraction': frac}


def word_shuffle(text: str, severity: int, rng: np.random.Generator) -> tuple[str, dict]:
    """Shuffle ~severity% of words to random positions (destroys local order)."""
    frac = SEVERITY_FRACTION[severity]
    words = text.split()
    n_shuf = max(2, int(round(len(words) * frac))) if len(words) >= 2 else 0
    if n_shuf == 0:
        return text, {'n_perturbed': 0}
    # Pick n_shuf positions, then permute the words at those positions.
    pick = sorted(rng.choice(len(words), size=min(n_shuf, len(words)), replace=False))
    picked_words = [words[i] for i in pick]
    perm = rng.permutation(len(picked_words))
    for j, i in enumerate(pick):
        words[i] = picked_words[perm[j]]
    return ' '.join(words), {'n_perturbed': len(pick), 'severity_fraction': frac}


def keyword_removal(text: str, severity: int, rng: np.random.Generator,
                     idf_scores: dict = None) -> tuple[str, dict]:
    """Remove the top-K highest-IDF words from the text, where K scales with severity.

    severity 1 -> 1 keyword, 3 -> 3, 5 -> 5

    A high-IDF word is a content word that appears in few documents (e.g.
    'stocks', 'satellite', 'election'). Removing these damages topical signal.
    idf_scores must be a {word -> idf_score} dict, precomputed from the train
    set, passed in at call time."""
    if idf_scores is None:
        raise ValueError("idf_scores dict must be provided")
    K_BY_SEV = {1: 1, 3: 3, 5: 5}
    K = K_BY_SEV[severity]
    words = text.split()
    if not words:
        return text, {'n_perturbed': 0}
    # Rank each word position by IDF (high = remove first); fall back to 0 for OOV
    scored = []
    for i, w in enumerate(words):
        bare = w.lower().strip(string.punctuation)
        scored.append((idf_scores.get(bare, 0.0), i))
    # Pick the K positions with highest IDF (ties broken by index)
    scored.sort(reverse=True)
    drop_positions = set()
    for score, i in scored:
        if score == 0.0: break
        drop_positions.add(i)
        if len(drop_positions) >= K:
            break
    if not drop_positions:
        return text, {'n_perturbed': 0}
    out = [w for i, w in enumerate(words) if i not in drop_positions]
    return ' '.join(out), {'n_perturbed': len(drop_positions), 'k': K}


def front_truncation(text: str, severity: int, rng: np.random.Generator) -> tuple[str, dict]:
    """Drop the first N% of words (forces the classifier to read past the lede).
    News articles tend to put the topic in the lede so this removes topic signal."""
    DROP_BY_SEV = {1: 0.20, 3: 0.50, 5: 0.75}
    drop_frac = DROP_BY_SEV[severity]
    words = text.split()
    if not words:
        return text, {'n_perturbed': 0}
    n_drop = max(1, int(round(len(words) * drop_frac)))
    if n_drop >= len(words):
        n_drop = len(words) - 1  # keep at least one word so output non-empty
    out = words[n_drop:]
    return ' '.join(out), {'n_perturbed': n_drop, 'drop_fraction': drop_frac}


_DIACRITIC_MAP = {
    'a': 'áàâäãå', 'e': 'éèêë', 'i': 'íìîï', 'o': 'óòôöõ', 'u': 'úùûü',
    'n': 'ñ', 'c': 'ç', 's': 'š', 'z': 'ž', 'y': 'ýÿ',
    'A': 'ÁÀÂÄÃÅ', 'E': 'ÉÈÊË', 'I': 'ÍÌÎÏ', 'O': 'ÓÒÔÖÕ', 'U': 'ÚÙÛÜ',
    'N': 'Ñ', 'C': 'Ç', 'S': 'Š', 'Z': 'Ž', 'Y': 'ÝŸ',
}


def diacritic_substitution(text: str, severity: int, rng: np.random.Generator) -> tuple[str, dict]:
    """Replace ~severity% of eligible ASCII letters with diacritic variants.
    Breaks BPE tokenization without removing semantic meaning."""
    frac = SEVERITY_FRACTION[severity]
    chars = list(text)
    eligible_idx = [i for i, c in enumerate(chars) if c in _DIACRITIC_MAP]
    if not eligible_idx:
        return text, {'n_perturbed': 0}
    n_perturb = max(1, int(round(len(eligible_idx) * frac)))
    pick = rng.choice(len(eligible_idx), size=n_perturb, replace=False)
    for k in pick:
        i = eligible_idx[k]
        variants = _DIACRITIC_MAP[chars[i]]
        chars[i] = variants[int(rng.integers(0, len(variants)))]
    return ''.join(chars), {'n_perturbed': n_perturb, 'severity_fraction': frac}


CORRUPTIONS = {
    'butter_fingers':         butter_fingers,
    'random_casing':          random_casing,
    'word_swap':              word_swap,
    'synonym_substitution':   synonym_substitution,
    'word_deletion':          word_deletion,
    'word_shuffle':           word_shuffle,
    'keyword_removal':        keyword_removal,
    'front_truncation':       front_truncation,
    'diacritic_substitution': diacritic_substitution,
}
