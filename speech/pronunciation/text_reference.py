"""Versioned English text-only dictionary baseline; no audio or corpus phones."""
import re
import unicodedata

VERSION = 'cmudict-first-entry-text-v1'


def load_dictionary(path):
    entries = {}
    for line in path.read_text().splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        # Use the unnumbered entry only; alternatives require a separate protocol.
        if not re.search(r'\(\d+\)$', fields[0]):
            if fields[0] in entries:
                raise ValueError('Duplicate dictionary base entry')
            entries[fields[0]] = fields[1:]
    return entries


def text_to_phones(text, dictionary):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('reference_text is required')
    normalized = unicodedata.normalize('NFKC', text).replace('’', "'").lower()
    words = re.findall(r"[a-z]+(?:'[a-z]+)*|[^\W_]+", normalized)
    if not words:
        raise ValueError('reference_text has no words')
    phones, unknown = [], []
    for word in words:
        if word not in dictionary:
            unknown.append(word)
            # Preserves missing-reference flags in phone_evidence; never drop words.
            phones.append('<UNKNOWN:' + word + '>')
        else:
            phones.extend(dictionary[word])
    return phones, dict(words=len(words), unknown_words=unknown,
                       complete=not unknown, normalization=VERSION)
