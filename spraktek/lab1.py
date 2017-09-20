import io
import regex as re
import pickle
import os
import math
from scipy import linalg, mat, dot
import numpy as np


def index(file, fileName):
    wordCount = 0
    dictionary = {}
    file1 = io.open('Selma/' + file).read().lower()
    pattern = '\p{L}+'
    regex = re.compile(pattern)
    for match in regex.finditer(file1):
        wordCount = wordCount + 1
        if match.group() not in dictionary:
            dictionary[match.group()] = []
            wordset = dictionary[match.group()]
            wordset.append(match.start())
            # ordNils[match.group()].add[match.start()]
        else:
            wordset = dictionary[match.group()]
            wordset.append(match.start())
            # ordNils[match.group()].add[match.start()]
    pickle.dump(dictionary, open('lab1files/' + fileName[0] + ".p", "wb"))
    return wordCount


def masterIndexer(fileName, wordIndex):
    title = fileName[0]
    file = pickle.load(open('lab1files/' + title + '.p', 'rb'))
    for word in file:
        if word not in wordIndex:
            wordIndex[word] = {title: file[word]}
        else:
            (wordIndex[word])[title] = file[word]
    return wordIndex


def cleanUp():
    for file in os.listdir('./'):
        if file.endswith('.idx'):
            os.remove(file)


def get_files(dir, suffix):
    """
  Returns all the files in a folder ending with suffix
  :param dir:
  :param suffix:
  :return: the list of file names
  """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files


def compare(fileList, all_tfidf):
    #build vectors:
    listOfFreqLists = {}
    maxSimilarity = 0
    for file in fileList:
        fileName = file.split('.')[0]
        freqList = []
        for word in all_tfidf[fileName]:
            freqList.append(all_tfidf[fileName][word])
        listOfFreqLists[fileName] = freqList
    for file in fileList:
        fileName1 = file.split('.')[0]
        for file in fileList:
            fileName2 = file.split('.')[0]
            if fileName1 != fileName2:
                matrix = mat([np.array(listOfFreqLists[fileName1]), np.array(listOfFreqLists[fileName2])])
                similarity = dot(matrix[0], matrix[1].T) / np.linalg.norm(matrix[0]) / np.linalg.norm(matrix[1])
                print(fileName1 + ' -> ' + fileName2)
                print(similarity)
                if similarity > maxSimilarity:
                    maxPair = fileName1 + ' -> ' + fileName2
                    maxSimilarity = similarity
    print('the most similar books are: ' + maxPair + ' with a score of:')
    print(maxSimilarity)



def calcTFIDF(fileList, wordList, totalWords):
    all_tfidf = {}
    for file in fileList:
        tf = {}
        idf= {}
        tfidf = {}
        fileName = file.split('.')[0]
        for word in wordList:
            if fileName not in wordList[word]:
                tf[word] = 0
                tfidf[word] = 0
            else:
                tf[word] = len(wordList[word][fileName]) / totalWords[fileName]
                idf[word] = math.log10(len(wordList[word]) / 9)
                tfidf[word] = -tf[word] * idf[word]
        #pickle.dump(tfidf, open(fileName + '_tf-idf.p', 'wb'))
        all_tfidf[fileName] = tfidf
    return all_tfidf

def start():
    fileList = get_files('Selma/', 'txt')
    wordIndex = {}
    totalWords = {}
    for file in fileList:
        fileName = file.split('.')
        totalWords[fileName[0]] = index(file, fileName)
    for file in fileList:
        fileName = file.split('.')
        wordIndex = masterIndexer(fileName, wordIndex)
    print(totalWords)
    all_tfidf = calcTFIDF(fileList, wordIndex, totalWords)
    compare(fileList, all_tfidf)

start()