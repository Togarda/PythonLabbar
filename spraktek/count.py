"""
A word counting program
Usage: python count.py < corpus.txt
"""
__author__ = "Pierre Nugues"

import sys

import regex as re


def tokenize(text):
    words = re.findall('[\S]+', text)
    return words


def count_unigrams(words):
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency


if __name__ == '__main__':
    text = sys.stdin.read().lower()
    words = tokenize(text)
    wordFreqMap = []
    frequency = count_unigrams(words)
    totalWords = len(words)
    sentence = "att han sa det".lower().split(' ')
    print(sentence)
    for word in sentence:
        wordFreqMap.append((word, frequency[word] / totalWords))
    print(wordFreqMap)

    sentenceFrequency = 1
    geometricMean = 0
    for word in wordFreqMap:
        sentenceFrequency = sentenceFrequency * word[1]
        geometricMean += word[1]
    print(sentenceFrequency)
    print("geometric mean: " + str(geometricMean/len(wordFreqMap)))
    print("perplexity: " + str(1/(geometricMean/len(wordFreqMap))))


    #for word in sorted(frequency.keys(), key=frequency.get, reverse=False):
        #print(word, '\t', frequency[word])