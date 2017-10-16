#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 17:54:00 2017

@author: draganadzaip
"""

import sklearn
import conll
import transition


def extract(stack, queue, graph, feature_name, sentences, featurenbr):
    X_l = []
    y_l = ''

    if featurenbr == 1:
        if not stack:
            X_l.append('nil')
            X_l.append('nil')
            X_l.append(queue[0]['form'])
            X_l.append(queue[0]['postag'])
        else:
            X_l.append(stack[0]['form'])
            X_l.append(stack[0]['postag'])
            X_l.append(queue[0]['form'])
            X_l.append(queue[0]['postag'])


    elif featurenbr == 2:
        if not stack:
            for i in range(4):
                X_l.append('nil')
            X_l.append(queue[0]['postag'])
            try:
                X_l.append(queue[1]['postag'])
            except:
                X_l.append('nil')
            X_l.append(queue[0]['form'])
            try:
                X_l.append(queue[1]['form'])
            except:
                X_l.append('nil')

        else:
            X_l.append(stack[0]['postag'])
            try:
                X_l.append(stack[1]['postag'])
            except:
                X_l.append('nil')
            X_l.append(stack[0]['form'])
            try:
                X_l.append(stack[1]['form'])
            except:
                X_l.append('nil')
            X_l.append(queue[0]['postag'])
            try:
                X_l.append(queue[1]['postag'])
            except:
                X_l.append('nil')
            X_l.append(queue[0]['form'])
            try:
                X_l.append(queue[1]['form'])
            except:
                X_l.append('nil')

    elif featurenbr == 3:
        if not stack:
            for i in range(6):
                X_l.append('nil')
            X_l.append(queue[0]['postag'])
            try:
                X_l.append(queue[1]['postag'])
            except:
                X_l.append('nil')
            try:
                X_l.append(queue[2]['postag'])
            except:
                X_l.append('nil')
            X_l.append(queue[0]['form'])
            try:
                X_l.append(queue[1]['form'])
            except:
                X_l.append('nil')
            try:
                X_l.append(queue[2]['form'])
            except:
                X_l.append('nil')
        else:
            X_l.append(stack[0]['postag'])
            try:
                X_l.append(stack[1]['postag'])
            except:
                X_l.append('nil')
            try:
                X_l.append(stack[2]['postag'])
            except:
                X_l.append('nil')
            X_l.append(stack[0]['form'])
            try:
                X_l.append(stack[1]['form'])
            except:
                X_l.append('nil')
            try:
                X_l.append(stack[2]['form'])
            except:
                X_l.append('nil')
            X_l.append(queue[0]['postag'])
            try:
                X_l.append(queue[1]['postag'])
            except:
                X_l.append('nil')
            try:
                X_l.append(queue[2]['postag'])
            except:
                X_l.append('nil')
            X_l.append(queue[0]['form'])
            try:
                X_l.append(queue[1]['form'])
            except:
                X_l.append('nil')
            try:
                X_l.append(queue[2]['form'])
            except:
                X_l.append('nil')

    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        y_l = 'ra' + deprel
    # Left arc
    elif stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        y_l = 'la' + deprel
    # Reduce
    elif stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                y_l = 're'
    else:
        y_l = 'sh'

    X_l.append(transition.can_reduce(stack, graph))
    X_l.append(transition.can_leftarc(stack, graph))

    X = dict(zip(feature_name, X_l))

    return X, y_l
