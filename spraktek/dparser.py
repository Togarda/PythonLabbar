"""
Gold standard parser
"""

#import transition
import conll
import features
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer


def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'


def encode_classes(y_symbols):
    """
    Encode the classes as numbers
    :param y_symbols:
    :return: the y vector and the lookup dictionaries
    """
    # We extract the chunk names
    classes = sorted(list(set(y_symbols)))
    """
    Results in:
    ['B-ADJP', 'B-ADVP', 'B-CONJP', 'B-INTJ', 'B-LST', 'B-NP', 'B-PP',
    'B-PRT', 'B-SBAR', 'B-UCP', 'B-VP', 'I-ADJP', 'I-ADVP', 'I-CONJP',
    'I-INTJ', 'I-NP', 'I-PP', 'I-PRT', 'I-SBAR', 'I-UCP', 'I-VP', 'O']
    """
    # We assign each name a number
    dict_classes = dict(enumerate(classes))
    """
    Results in:
    {0: 'B-ADJP', 1: 'B-ADVP', 2: 'B-CONJP', 3: 'B-INTJ', 4: 'B-LST',
    5: 'B-NP', 6: 'B-PP', 7: 'B-PRT', 8: 'B-SBAR', 9: 'B-UCP', 10: 'B-VP',
    11: 'I-ADJP', 12: 'I-ADVP', 13: 'I-CONJP', 14: 'I-INTJ',
    15: 'I-NP', 16: 'I-PP', 17: 'I-PRT', 18: 'I-SBAR',
    19: 'I-UCP', 20: 'I-VP', 21: 'O'}
    """

    # We build an inverted dictionary
    inv_dict_classes = {v: k for k, v in dict_classes.items()}
    """
    Results in:
    {'B-SBAR': 8, 'I-NP': 15, 'B-PP': 6, 'I-SBAR': 18, 'I-PP': 16, 'I-ADVP': 12,
    'I-INTJ': 14, 'I-PRT': 17, 'I-CONJP': 13, 'B-ADJP': 0, 'O': 21,
    'B-VP': 10, 'B-PRT': 7, 'B-ADVP': 1, 'B-LST': 4, 'I-UCP': 19,
    'I-VP': 20, 'B-NP': 5, 'I-ADJP': 11, 'B-CONJP': 2, 'B-INTJ': 3, 'B-UCP': 9}
    """

    # We convert y_symbols into a numerical vector
    y = [inv_dict_classes[i] for i in y_symbols]
    return y, dict_classes, inv_dict_classes


def parse_ml(stack, queue, graph, trans):
    if stack and trans[:2] == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'
    elif stack and trans[:2] == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'
    elif stack and trans[:2] == 're':
        stack, queue, graph = transition.reduce(stack, queue, graph, trans[3:])
        return stack, queue, graph, 're'
    else:
        stack, queue, graph = transition.shift(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'sh'


if __name__ == '__main__':

    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    feature_1 = ['word_stack', 'pos_stack', 'word_queue', 'pos_queue', 'can_re',
                 'can_la']

    feature_2 = ['word_stack[0]', 'word_stack[1]', 'pos_stack[0]', 'pos_stack[1]',
                 'word_queue[0]', 'word_queue[1]', 'pos_queue[0]', 'pos_queue[1]', 'can_re',
                 'can_la']

    feature_3 = ['pos_stack[0]', 'pos_stack[1]', 'pos_stack[2]', 'word_stack[0]', 'word_stack[1]',
                 'word_stack[2]', 'pos_queue[0]', 'pos_queue[1]', 'pos_queue[2]', 'word_queue[0]',
                 'word_queue[1]', 'word_queue[2]', 'can_re', 'can_la']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    sentences_test = conll.read_sentences(test_file)
    formatted_corpus_test = conll.split_rows(sentences_test, column_names_2006)

    x_list = []
    y_list = []

    sent_cnt = 0
    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []

        x_templist = []
        y_templist = []

        while queue:
            current_dictX, current_Y = features.extract(stack, queue, graph, feature_1, sentence, 1)
            stack, queue, graph, trans = reference(stack, queue, graph)
            transitions.append(trans)

            x_templist.append(current_dictX)
            y_templist.append(current_Y)

        stack, graph = transition.empty_stack(stack, graph)

        for word in sentence:
            word['head'] = graph['heads'][word['id']]

        x_list.extend(x_temp_list)
        y_list.extend(y_temp_list)

    print("Encoding the features and classes...")
    # Vectorize the feature matrix and carry out a one-hot encoding
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(x_list)
    # The statement below will swallow a considerable memory
    # X = vec.fit_transform(X_dict).toarray()
    # print(vec.get_feature_names())

    y, nbr_to_class, classes_to_nbr = encode_classes(y_list)

    print("Training the model...")
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    model = classifier.fit(X, y)
    print(model)
    print('Predicting')

    # print(transitions)
    # print(graph)


    x_list = []
    y_list = []
    graph_list = []

    sent_cnt = 0
    for sentence in formatted_corpus_test:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []

        x_templist = []
        y_templist = []

        while queue:
            current_dictX, current_Y = features.extract(stack, queue, graph, feature_1, sentence, 1)

            X = vec.transform(current_dictX)
            trans = classifier.predict(X)
            stack, queue, graph, trans = parse_ml(stack, queue, graph, trans)
            y_templist.append(trans)

        graph_list.append(graph)
        stack, graph = transition.empty_stack(stack, graph)
        y_list.append(y_templist)

    for i in range(len(formatted_corpus_test)):
        current_graph = graph_list[i]
        for word in formatted_corpus_test[i]:
            word['head'] = current_graph['heads'][word['id']]
            word['deprel'] = current_graph['deprels'][word['id']]
            word['phead'] = '_'
            word['pdeprel'] = '_'

    f_out = open('output', 'w')
    for sentence in formatted_corpus_test:
        for row in sentence[1:]:
            f_out.write(row['id'] + '\t' + row['form'] + '\t' + row['lemma'] + '\t' + row['cpostag'] + '\t' + row[
                'postag'] + '\t' + row['feats'] + '\t' +
                        row['head'] + '\t' + row['deprel'] + '\t' + row['phead'] + '\t' + row['pdeprel'] + '\n')
        f_out.write('\n')
    f_out.close()

