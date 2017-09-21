"""
A word counting program
Usage: python count.py < corpus.txt
"""
__author__ = "Pierre Nugues"

import sys
import math
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

def count_bigrams(words):
    bigrams = [tuple(words[inx:inx + 2])
               for inx in range(len(words) - 1)]
    frequencies = {}
    for bigram in bigrams:
        if bigram in frequencies:
            frequencies[bigram] += 1
        else:
            frequencies[bigram] = 1
    return frequencies

def unigram_model(sentence):
    #init
    sentenceFrequency = 1
    geometricMean = 1
    wordFreqMap = []
    frequency = count_unigrams(words)
    sentence_list = sentence.split(' ')
    for word in sentence_list:
        wordFreqMap.append((word, frequency[word] / totalWords))
    # print(wordFreqMap)
    print("""Unigram model
==========================================
wi  C(wi)   #words  P(wi)
==========================================""")
    for word in wordFreqMap:
        sentenceFrequency = sentenceFrequency * word[1]
        print(word[0], frequency[word[0]], totalWords, word[1])
        geometricMean *= word[1]
    geometricMean = (geometricMean / len(wordFreqMap)) ** (1 / len(sentence_list))
    print('==========================================')
    print('Prob. unigrams: ', sentenceFrequency)
    print("Geometric mean prob: " + str(geometricMean))
    print('Entropy rate: ', calc_entropy(sentenceFrequency, len(sentence_list)))
    print("Perplexity: " + str(1 / (geometricMean)))

def bigram_model(sentence):
    #init
    sentenceFrequency = 1
    geometricMean = 1
    wordFreqMap = {}
    sentence_list = []
    frequency_bi = count_bigrams(words)
    frequency_uni = count_unigrams(words)
    sentence = '<s> ' + sentence + ' </s>'
    sentence_list = sentence.split(' ')

    for i in range(0, (len(sentence_list)-1)):
        key = (sentence_list[i], sentence_list[i+1])
        if key in frequency_bi.keys():
            wordFreqMap[key] = (frequency_bi[key]/totalWords)
        else:
            wordFreqMap[key] = (frequency_uni[key[1]]/totalWords)

    #Print
    print("""Bigram model
==========================================
wi  wi+1    C(wi)   #words  P(wi)
==========================================""")

    print(wordFreqMap)
    for word in wordFreqMap:
        if word in frequency_bi:
            sentenceFrequency = sentenceFrequency * wordFreqMap[word]
            print(word, frequency_bi[word], frequency_uni[word[0]], frequency_bi[word] / (totalWords -1))
        else:
            sentenceFrequency = sentenceFrequency * frequency_uni[word[1]]
            print(word, 0, frequency_uni[word[0]], '*backoff: ', sentenceFrequency)
        #sentenceFrequency = sentenceFrequency * word[1]
        #print(word[0][0],word[0][1], frequency_bi[word[0]], totalWords, word[1])
        #geometricMean *= word[1]
    geometricMean = (geometricMean / len(wordFreqMap)) ** (1 / len(sentence))
    print('==========================================')
    print('Prob. bigrams: ', sentenceFrequency)
    print("Geometric mean prob: " + str(geometricMean))
    print('Entropy rate: ', calc_entropy(sentenceFrequency, len(sentence)))
    print("Perplexity: " + str(1 / (geometricMean)))

def calc_entropy(sentence_prob, length):
    return -1/length*math.log(sentence_prob)

if __name__ == '__main__':
    text = sys.stdin.read().lower()
    words = tokenize(text)
    totalWords = len(words)
    sentence = "det var en gång en katt som hette Nils".lower()
    #unigram_model(sentence)
    bigram_model(sentence)

    #for word in sorted(frequency.keys(), key=frequency.get, reverse=False):
        #print(word, '\t', frequency[word])