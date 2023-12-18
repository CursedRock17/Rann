import os
import glob as glob
import string_assistance as string_


def findFiles(path):
    return glob.glob(path)


def splitCode(filename):
    # Read the file and split the file into lines
    lines = open('data/%s.txt' % (filename), encoding='utf-8').read().strip()
    lines = (["def" + func for func in lines.split("def")])
    return lines


def splitLabels(filename):
    lines = open('data/%s.txt' % (filename), encoding='utf-8').read().strip().split('\n')
    lines = [[string_.normalizeString(string) for string in line.split('\t')] for line in lines]
    return lines


def organizePairs(code_filename, label_filename):
    pairs = [splitCode(code_filename), splitLabels(label_filename)]
    return pairs


pairs = organizePairs("code", "codeLabels")
