import unicodedata
import re


# Turn unicode to ASCII
def unicodeToAscii(s):
    return ''.join(
            char for char in unicodedata.normalize('NFD', s)
            if unicodedata.category(char) != 'Mn'
    )


# Remove all non-letter characters and lowercase and trim
def normalizeString(string):
    string = unicodeToAscii(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z!?]+", r" ", string)
    return string.strip()
