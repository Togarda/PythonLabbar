"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows if row[0] != '#']
        sentence = start + sentence
        new_sentences.append(sentence)


    svPairs = 0
    svoTriples = 0

    dictionary = {}

    for sentence in new_sentences:
        for word in sentence:
            if word['deprel'] == 'SS':
                firstword = word['form'].lower()
                head = word['head']
                svPairs += 1
                for otherwords in sentence:
                    if otherwords['id'] == head:
                        tuple = (firstword, otherwords['form'].lower())
                        if tuple in dictionary:
                            dictionary[tuple] += 1
                        else:
                            dictionary[tuple] = 1

    sortedPairs = sorted(dictionary, key=dictionary.get, reverse=True)[:5]
    print("5 most common subject-verb pairs: ")
    for key in sortedPairs:
        print(key[0] + "-" + key[1] + ": ", dictionary[key])
    print("---------------------------------------------------------")
    print("total amount of subject-verb pairs:")
    print(svPairs)
    print("---------------------------------------------------------")

    tripledict = {}
    triple_par = 0
    for sentence in new_sentences:
        for word in sentence:
            if word['deprel'] == 'SS':
                firstword = word['form'].lower()
                head = word['head']
                for verb in sentence:
                    if verb['id'] == head:
                        secondword = verb['form'].lower()
                        for possibleObject in sentence:
                            if possibleObject['head'] == head and possibleObject['deprel'] == 'OO':
                                thirdword = possibleObject['form'].lower()
                                triple_par += 1
                                tuple = (firstword, secondword, thirdword)
                                if tuple in tripledict:
                                    tripledict[tuple] += 1
                                else:
                                    tripledict[tuple] = 1

    nyArray = sorted(tripledict, key=tripledict.get, reverse=True)[:5]
    for key in nyArray:
        print(key[0] + "-" + key[1] + "-" + key[2] + ": ", tripledict[key])
    print("---------------------------------------------------------")
    print("total amount of subject-verb-object triples:")
    print(triple_par)
    return new_sentences


def save(file, formatted_corpus, column_names):
    f_out = open(file, 'w')
    for sentence in formatted_corpus:
        for row in sentence[1:]:
            # print(row, flush=True)
            for col in column_names[:-1]:
                if col in row:
                    f_out.write(row[col] + '\t')
                else:
                    f_out.write('_\t')
            col = column_names[-1]
            if col in row:
                f_out.write(row[col] + '\n')
            else:
                f_out.write('_\n')
        f_out.write('\n')
    f_out.close()


if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    train_file = 'swedish_talbanken05_train.conll'
    # train_file = 'test_x'
    test_file = 'swedish_talbanken05_test.conll'

    sentences = read_sentences(train_file)
    formatted_corpus = split_rows(sentences, column_names_2006)
    print(train_file, len(formatted_corpus))
    print(formatted_corpus[0])

    column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

    files = get_files('../../corpus/ud-treebanks-v1.3/', 'train.conllu')
    for train_file in files:
        sentences = read_sentences(train_file)
        formatted_corpus = split_rows(sentences, column_names_u)
        print(train_file, len(formatted_corpus))
        print(formatted_corpus[0])