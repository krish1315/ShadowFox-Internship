import re
from collections import defaultdict, Counter
from spellchecker import SpellChecker

CORPUS_FILE = "corpus.txt"   # large real-world training file (e.g. auto_correct.txt)


def load_corpus(path, fallback_text=""):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except FileNotFoundError:
        return fallback_text


# Small fallback corpus used only if CORPUS_FILE is missing, so the script never crashes.
FALLBACK_CORPUS = """
I love to play cricket in the evening with my friends.
I want to learn how to code in python every day.
The weather is very nice today and I want to go for a walk.
Machine learning is a fascinating field of computer science.
"""

CORPUS = load_corpus(CORPUS_FILE, FALLBACK_CORPUS)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def tokenize(text):
    return clean_text(text).split()


class NgramModel:
    def __init__(self, n):
        self.n = n
        self.contexts = defaultdict(Counter)

    def train(self, tokens):
        for i in range(len(tokens) - self.n + 1):
            gram = tokens[i:i + self.n]
            context = tuple(gram[:-1])
            next_word = gram[-1]
            self.contexts[context][next_word] += 1

    def predict(self, context, top_k=3):
        context = tuple(context[-(self.n - 1):])
        if context not in self.contexts:
            return []
        counter = self.contexts[context]
        total = sum(counter.values())
        ranked = counter.most_common(top_k)
        return [(word, round(count / total, 2)) for word, count in ranked]


class NextWordPredictor:
    def __init__(self, corpus, max_n=3):
        self.vocab = set()
        self.models = {}
        tokens = tokenize(corpus)
        self.vocab.update(tokens)
        for n in range(2, max_n + 1):
            model = NgramModel(n)
            model.train(tokens)
            self.models[n] = model

    def predict_next(self, sentence, top_k=3):
        tokens = tokenize(sentence)
        for n in sorted(self.models.keys(), reverse=True):
            context_size = n - 1
            if len(tokens) >= context_size:
                context = tokens[-context_size:]
                suggestions = self.models[n].predict(context, top_k)
                if suggestions:
                    return suggestions
        return []


class SpellAutocorrector:
    def __init__(self, extra_vocab=None):
        self.spell = SpellChecker()
        if extra_vocab:
            self.spell.word_frequency.load_words(extra_vocab)

    def correct_word(self, word):
        core = re.sub(r"[^A-Za-z']", "", word).strip("'\"")
        if not core:
            return word
        if core.lower() not in self.spell.unknown([core]):
            return word
        fix = self.spell.correction(core)
        return word.replace(core, fix) if fix else word


class SmartKeyboard:
    def __init__(self, corpus):
        self.predictor = NextWordPredictor(corpus)
        self.autocorrector = SpellAutocorrector(extra_vocab=self.predictor.vocab)

    def process(self, typed_text):
        words = typed_text.strip().split()
        if not words:
            return "", []
        words[-1] = self.autocorrector.correct_word(words[-1])
        corrected_text = " ".join(words)
        suggestions = self.predictor.predict_next(corrected_text)
        return corrected_text, suggestions


if __name__ == "__main__":
    keyboard = SmartKeyboard(CORPUS)

    test_inputs = [
        "to be or not to",
        "I love",
        "romeo and",
        "i want to lern",
    ]

    for typed in test_inputs:
        corrected_text, suggestions = keyboard.process(typed)
        print(f"You typed:      {typed}")
        print(f"Autocorrected:  {corrected_text}")
        print(f"Next word tips: {suggestions}")
        print("-" * 55)
