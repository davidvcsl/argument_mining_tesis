#!/usr/bin/python
#Usage: python RunModel.py modelPath inputPath"
from __future__ import print_function
import nltk
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys, os

if len(sys.argv) < 3:
    print("Usage: python RunModel.py modelPath inputPath")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]

with open(inputPath, 'r') as f:
    text = f.read()
    if len(text.split("\t"))>1:
        os.system('python util/conll_to_txt.py --input={} --output={}_txt --target_column={}'.format(inputPath,inputPath,1))
        with open(inputPath + "_txt", 'w') as newf:
            text = newf.read() 

# :: Load the model ::
lstmModel = BiLSTM()
lstmModel.loadModel(modelPath)

paragraphs = filter(lambda x: x != '', text.splitlines())
# :: Prepare the input ::
sentences = [{'tokens': nltk.word_tokenize(par)} for par in paragraphs]
addCharInformation(sentences)
addCasingInformation(sentences)

dataMatrix = createMatrices(sentences, lstmModel.mappings)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


# :: Output to stdout ::
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']
    tokenTags = tags[sentenceIdx]
    for tokenIdx in range(len(tokens)):
        print("{}\t{}".format(tokens[tokenIdx], tokenTags[tokenIdx]))
    print("")

