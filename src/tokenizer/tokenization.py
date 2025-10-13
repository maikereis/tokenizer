import re
from typing import Callable

from nltk.corpus import stopwords

from tokenizer.constants import ADDRESS_TYPE, CUSTOM_STOPWORDS

STOPWORDS = set(stopwords.words("portuguese"))
TOKEN_PATTERN = re.compile(r"\d{5}-\d{3}|S/N|\d+|[A-Za-zÀ-ÿ]+", flags=re.IGNORECASE)
UFS = ["pa", "ma", "pi", "al", "rs", "go"]


def tokenize_address_strategy(text: str, n=3, pre_processing: Callable = None):
    # 1. Lowercase and normalize (remove accentuation)
    text_without_accents = pre_processing(text)

    # 2. Tokenize directly using regex
    tokens = TOKEN_PATTERN.findall(text_without_accents)

    # 3. Remove stopwords
    filtered = [t for t in tokens if t not in STOPWORDS]
    filtered = [t for t in tokens if t not in CUSTOM_STOPWORDS]

    # 4. Exception: reinclude "para" if it was in the original
    if "pará" in text.lower():
        if "para" not in filtered:
            filtered.append("para")

    strong_tokens = set()
    weak_tokens = set()

    # CEP (ex: 68400000 ou 68400-000)
    ceps = [t for t in filtered if re.fullmatch(r"\d{5}-?\d{3}", t)]
    for cep in ceps:
        cep_num = cep.replace("-", "")
        strong_tokens.add(cep_num[:5])  # prefixo
        strong_tokens.add(cep_num[:3])  # mais geral

    # numbers (ex: 713, SN)
    numeros = [t for t in filtered if re.fullmatch(r"\d+|sn|s/n", t)]
    for num in numeros:
        strong_tokens.add(num)

    # state UF
    if (tokens[0] in UFS) and (tokens[0] not in strong_tokens):
        strong_tokens.add(tokens[0])

    # if there is a street type + number
    for i, tok in enumerate(filtered[:-1]):
        if tok in ADDRESS_TYPE and re.fullmatch(r"\d+|sn", filtered[i + 1]):
            strong_tokens.add(f"{tok} {filtered[i+1]}")

    # Add highway/road markers (BR, KM)
    for i, tok in enumerate(filtered[:-1]):
        # Match BR patterns: BR 230, BR-010
        if re.fullmatch(r"br|br-", tok):
            next_tok = filtered[i + 1]
            if re.fullmatch(r"\d{1,3}", next_tok):
                strong_tokens.add(f"br {next_tok}")

        # Match KM patterns: KM 02, KM 2, KM 13
        if tok.lower() == "km":
            next_tok = filtered[i + 1]
            if re.fullmatch(r"\d{1,3}", next_tok):
                strong_tokens.add(f"km {next_tok}")

    # trigrams
    for token in filtered:
        if len(token) >= n:
            for i in range(0, len(token) - n + 1, n):
                weak_tokens.add(token[i : i + n])

    return list(strong_tokens.union(weak_tokens))
