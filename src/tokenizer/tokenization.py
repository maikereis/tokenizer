import re
from typing import Callable

from nltk.corpus import stopwords

from tokenizer.constants import ADDRESS_TYPE, CUSTOM_STOPWORDS

STOPWORDS = set(stopwords.words("portuguese"))
TOKEN_PATTERN = re.compile(r"\d{5}-\d{3}|S/N|\d+|[A-Za-zÀ-ÿ]+", flags=re.IGNORECASE)
UFS = ["pa", "ma", "pi", "al", "rs", "go"]


class AddressTokenizer:
    def __init__(self, n: int = 3, pre_processing: Callable[[str], str] = None):
        """
        n: size of weak token substrings
        pre_processing: callable function to preprocess text (e.g., normalize accents)
        """
        self.n = n
        self.pre_processing = pre_processing or (
            lambda x: x
        )  # default: no preprocessing

    def __call__(self, text: str) -> list[str]:
        # 1. Preprocess
        text_without_accents = self.pre_processing(text)

        # 2. Tokenize using regex
        tokens = TOKEN_PATTERN.findall(text_without_accents)

        # 3. Remove stopwords
        filtered = [t for t in tokens if t not in STOPWORDS]
        filtered = [t for t in filtered if t not in CUSTOM_STOPWORDS]

        # 4. Exception: reinclude "para" if "pará" was in original
        if "pará" in text.lower() and "para" not in filtered:
            filtered.append("para")

        strong_tokens = set()
        weak_tokens = set()

        # CEP
        ceps = [t for t in filtered if re.fullmatch(r"\d{5}-?\d{3}", t)]
        for cep in ceps:
            cep_num = cep.replace("-", "")
            strong_tokens.add(cep_num[:5])
            strong_tokens.add(cep_num[:3])

        # Numbers
        numeros = [t for t in filtered if re.fullmatch(r"\d+|sn|s/n", t)]
        for num in numeros:
            strong_tokens.add(num)

        # State UF
        if tokens and tokens[0] in UFS and tokens[0] not in strong_tokens:
            strong_tokens.add(tokens[0])

        # Street type + number
        for i, tok in enumerate(filtered[:-1]):
            if tok in ADDRESS_TYPE and re.fullmatch(r"\d+|sn", filtered[i + 1]):
                strong_tokens.add(f"{tok} {filtered[i+1]}")

        # Highway / KM markers
        for i, tok in enumerate(filtered[:-1]):
            # BR patterns
            if re.fullmatch(r"br|br-", tok.lower()):
                next_tok = filtered[i + 1]
                if re.fullmatch(r"\d{1,3}", next_tok):
                    strong_tokens.add(f"br {next_tok}")

            # KM patterns
            if tok.lower() == "km":
                next_tok = filtered[i + 1]
                if re.fullmatch(r"\d{1,3}", next_tok):
                    strong_tokens.add(f"km {next_tok}")

        # Weak token trigrams
        for token in filtered:
            if len(token) >= self.n:
                for i in range(0, len(token) - self.n + 1, self.n):
                    weak_tokens.add(token[i : i + self.n])

        return list(strong_tokens.union(weak_tokens))
