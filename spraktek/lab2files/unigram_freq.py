

import sys

import regex


def tokenize(text):
    words = regex.findall('\p{L}+', text)
    return words


def count_ngrams(words, n):
    ngrams = [tuple(words[inx:inx + n])
              for inx in range(len(words) - n + 1)]
    # "\t".join(words[inx:inx + n])
    frequencies = {}
    for ngram in ngrams:
        if ngram in frequencies:
            frequencies[ngram] += 1
        else:
            frequencies[ngram] = 1
    return frequencies


if __name__ == '__main__':
    n = 1
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    text = sys.stdin.read().lower()
    words = tokenize(text)
    # frequency = count_bigrams(words)
    frequency = count_ngrams(words, 1)
    for word in sorted(frequency, key=frequency.get, reverse=False):
        print(word)