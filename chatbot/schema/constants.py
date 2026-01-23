"""Constants for chatbot schema types."""

from typing import Literal

# Biological sex options
Sex = Literal["male", "female", "intersex", "unknown"]

# Pregnancy status options
Pregnancy = Literal["pregnant", "postpartum", "not_pregnant", "unknown"]

# Validation limits
MAX_SYMPTOMS_COUNT = 40
MAX_DISEASE_NAME_LENGTH = 120
MAX_AGE = 120
MAX_ONSET_DAYS = 3650
MAX_NAME_LENGTH = 150
MAX_BIO_LENGTH = 500
