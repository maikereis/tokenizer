from dataclasses import dataclass, asdict
import re
from typing import Callable, Dict, List

import nltk
from nltk.corpus import stopwords

from tokenizer.constants import ADDRESS_TYPE, CUSTOM_STOPWORDS

nltk.download("stopwords")


STOPWORDS = set(stopwords.words("portuguese"))
TOKEN_PATTERN = re.compile(r"\d{5}-\d{3}|S/N|\d+|[A-Za-zÀ-ÿ]+", flags=re.IGNORECASE)
UFS = ["pa", "ma", "pi", "al", "rs", "go"]

def identity(x):
    return x

@dataclass(frozen=True)
class Address:
    estado: str = ""
    cep: str = ""
    rua: str = ""
    numero: str = ""
    municipio: str = ""
    bairro: str = ""
    referencia: str = ""
    complemento: str = ""
    nome: str = ""
    latitude: str = ""
    longitude: str = ""

class AddressTokenizer:
    def __init__(
        self, 
        n: int = 3, 
        pre_processing: Callable[[str], str] = None,
        field_aware: bool = True,
        prefix_fields: set = None
    ):
        """
        n: size of weak token substrings
        pre_processing: callable function to preprocess text (e.g., normalize accents)
        field_aware: if True, prefix tokens with field names
        prefix_fields: set of field names to prefix (if None, prefix all fields)
        """
        self.n = n
        self.pre_processing = pre_processing or identity
        self.field_aware = field_aware
        # Default: prefix only fields prone to collisions
        self.prefix_fields = prefix_fields or {
            "estado",
            "cep",
            "rua",
            "numero",
            "municipio",
            "bairro",
            "referencia",
            "complemento",
            "nome",
            "latitude",
            "longitude",
        }

    def tokenize_text(self, text: str) -> List[str]:
        """Core tokenization logic for raw text."""
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

    def tokenize_field(self, field: str, value: str) -> List[str]:
        """Tokenize a single field value, optionally with field prefix."""
        if not value:
            return []
        
        base_tokens = self.tokenize_text(str(value))
        
        if self.field_aware and field in self.prefix_fields:
            return [f"{field}:{token}" for token in base_tokens]
        
        return base_tokens

    def tokenize_structured(self, address: Address) -> Dict[str, List[str]]:
        """Tokenize a structured Address object, returning tokens per field."""
        address_dict = asdict(address) if isinstance(address, Address) else address
        
        tokens = []

        for field, value in address_dict.items():
            if value:

                tokens.extend(self.tokenize_field(field, value))

        return tokens

    def __call__(self, input_data) -> List[str] | Dict[str, List[str]]:
        """
        Flexible tokenization:
        - str: tokenize as free text
        - Address/dict: tokenize as structured data
        """
        if isinstance(input_data, str):
            return self.tokenize_text(input_data)
        elif isinstance(input_data, (Address, dict)):
            return self.tokenize_structured(input_data)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")