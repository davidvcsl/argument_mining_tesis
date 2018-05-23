"""
A bidirectional LSTM with optional CRF and character-based presentation for NLP sequence tagging.

Author: Nils Reimers
License: CC BY-SA 3.0
"""

from __future__ import print_function

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

import os
import sys
import random
import time
import math
import numpy as np
import logging

from util.F1Validation import compute_f1_token_basis


import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

class BiLSTM:
    learning_rate_updates = {'sgd': {1: 0.1, 3:0.05, 5:0.01} } 
    verboseBuild = True
    model = None 
    epoch = 0 
    skipOneTokenSentences=True
    dataset = None
    embeddings = None
    labelKey = None
    writeOutput = True
    resultsOut = None
    modelSavePath = None
    maxCharLen = None
    
    params = {'miniBatchSize': 32,
              'dropout': [0.25, 0.25],
              'LSTM-Size': [100],
              'optimizer': 'nadam',
              'earlyStopping': -1,
              'clipvalue': 0,
              'clipnorm': 1 } #Default params


    def __init__(self, devEqualTest=False, params=None):
        if params != None:
            self.params.update(params)
        self.devEqualTest = devEqualTest
        logging.info("BiLSTM model initialized with parameters: %s" % str(self.params))
        
    def setMappings(self, embeddings, mappings):
        self.mappings = mappings
        self.embeddings = embeddings
        self.idx2Word = {v: k for k, v in self.mappings['tokens'].items()}

    def setTrainDataset(self, dataset, labelKey):
        self.dataset = dataset
        self.labelKey = labelKey
        self.label2Idx = self.mappings[labelKey]  
        self.idx2Label = {v: k for k, v in self.label2Idx.items()}
        self.mappings['label'] = self.mappings[labelKey]
        self.max_test_score = 0
        self.max_dev_score = 0
        self.max_scores = {'test': self.max_test_score,
                           'dev': self.max_dev_score}


    def trainModel(self):
        if self.model == None:
            self.buildModel()        
            
        trainMatrix = self.dataset['trainMatrix'] 
        self.epoch += 1
        
        if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:
            K.set_value(self.model.optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch])
            logging.info("Update Learning Rate to %f" % (K.get_value(self.model.optimizer.lr)))
        
        iterator = self.online_iterate_dataset(trainMatrix, self.labelKey) if self.params['miniBatchSize'] == 1 else self.batch_iterate_dataset(trainMatrix, self.labelKey)
        
        for batch in iterator: 
            labels = batch[0]
            nnInput = batch[1:]
            self.model.train_on_batch(nnInput, labels)   

    def predictLabels(self, sentences):
        if self.model == None:
            self.buildModel()
            
        predLabels = [None]*len(sentences)

        sentenceLengths = self.getSentenceLengths(sentences)

        for senLength, indices in sentenceLengths.items():
            if self.skipOneTokenSentences and senLength == 1:
                if 'O' in self.label2Idx:
                    dummyLabel = self.label2Idx['O']
                else:
                    dummyLabel = 0
                predictions = [[dummyLabel]] * len(indices) #Tag with dummy label
            else:          
                
                features = ['tokens']
                inputData = {name: [] for name in features}              
                
                for idx in indices:                    
                    for name in features:
                        inputData[name].append(sentences[idx][name])
                                                    
                for name in features:
                    inputData[name] = np.asarray(inputData[name])

                predictions = self.model.predict([inputData[name] for name in features], verbose=False)
                predictions = predictions.argmax(axis=-1) #Predict classes      

            predIdx = 0
            for idx in indices:
                predLabels[idx] = predictions[predIdx]
                sentences[idx]['label'] = predictions[predIdx]
                predIdx += 1   
        
        return predLabels
    
    
    # ------------ Some help functions to train on sentences -----------
    def online_iterate_dataset(self, dataset, labelKey): 
        idxRange = list(range(0, len(dataset)))
        random.shuffle(idxRange)
        
        for idx in idxRange:
                labels = []                
                features = ['tokens']
                
                labels = dataset[idx][labelKey]
                labels = [labels]
                labels = np.expand_dims(labels, -1)  
                    
                inputData = {}              
                for name in features:
                    inputData[name] = np.asarray([dataset[idx][name]])                 
                                    
                 
                yield [labels] + [inputData[name] for name in features] 
            
            
            
    def getSentenceLengths(self, sentences):
        sentenceLengths = {}
        for idx in range(len(sentences)):
            sentence = sentences[idx]['tokens']
            if len(sentence) not in sentenceLengths:
                sentenceLengths[len(sentence)] = []
            sentenceLengths[len(sentence)].append(idx)
        
        return sentenceLengths
            
    
    trainSentenceLengths = None
    trainSentenceLengthsKeys = None        
    def batch_iterate_dataset(self, dataset, labelKey):       
        if self.trainSentenceLengths == None:
            self.trainSentenceLengths = self.getSentenceLengths(dataset)
            self.trainSentenceLengthsKeys = list(self.trainSentenceLengths.keys())
            
        trainSentenceLengths = self.trainSentenceLengths
        trainSentenceLengthsKeys = self.trainSentenceLengthsKeys
        
        random.shuffle(trainSentenceLengthsKeys)
        for senLength in trainSentenceLengthsKeys:
            if self.skipOneTokenSentences and senLength == 1: #Skip 1 token sentences
                continue
            sentenceIndices = trainSentenceLengths[senLength]
            random.shuffle(sentenceIndices)
            sentenceCount = len(sentenceIndices)
            
            
            bins = int(math.ceil(sentenceCount/float(self.params['miniBatchSize'])))
            binSize = int(math.ceil(sentenceCount / float(bins)))
           
            numTrainExamples = 0
            for binNr in range(bins):
                tmpIndices = sentenceIndices[binNr*binSize:(binNr+1)*binSize]
                numTrainExamples += len(tmpIndices)
                
                
                labels = []
                features = ['tokens']
                inputData = {name: [] for name in features}

                for idx in tmpIndices:
                    labels.append(dataset[idx][labelKey])
                    
                    for name in features:
                        inputData[name].append(dataset[idx][name])                 
                                    
                labels = np.asarray(labels)
                labels = np.expand_dims(labels, -1)
                
                for name in features:
                    inputData[name] = np.asarray(inputData[name])
                 
                yield [labels] + [inputData[name] for name in features]   
                
            assert(numTrainExamples == sentenceCount) #Check that no sentence was missed 
            

    def buildModel(self):        
        params = self.params  
        
        embeddings = self.embeddings

        tokens = Sequential()
        tokens.add(Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1],  weights=[embeddings], trainable=False, name='token_emd'))
    
        layerIn = tokens.input
        layerOut = tokens.output

        # Add LSTMs
        cnt = 1
        for size in params['LSTM-Size']:
            if isinstance(params['dropout'], (list, tuple)):
                lstmLayer = Bidirectional(LSTM(size, return_sequences=True, dropout=params['dropout'][0], recurrent_dropout=params['dropout'][1]), name="main_LSTM_"+str(cnt))(layerOut)
            
            else:
                """ Naive dropout """
                lstmLayer = Bidirectional(LSTM(size, return_sequences=True), name="LSTM_"+str(cnt))(layerOut)
                
                if params['dropout'] > 0.0:
                    lstmLayer = TimeDistributed(Dropout(params['dropout']), name="dropout_"+str(cnt))(lstmLayer)
            
            cnt += 1
        

        # Softmax Decoder
        activationLayer = TimeDistributed(Dense(len(self.dataset['mappings'][self.labelKey]), activation='softmax'), name='softmax_output')(lstmLayer)
        lossFct = 'sparse_categorical_crossentropy'
       
        optimizerParams = {}
        if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
            optimizerParams['clipnorm'] = self.params['clipnorm']
        
        if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
            optimizerParams['clipvalue'] = self.params['clipvalue']
        
        if params['optimizer'].lower() == 'adam':
            opt = Adam(**optimizerParams)
        elif params['optimizer'].lower() == 'nadam':
            opt = Nadam(**optimizerParams)
        elif params['optimizer'].lower() == 'rmsprop': 
            opt = RMSprop(**optimizerParams)
        elif params['optimizer'].lower() == 'adadelta':
            opt = Adadelta(**optimizerParams)
        elif params['optimizer'].lower() == 'adagrad':
            opt = Adagrad(**optimizerParams)
        elif params['optimizer'].lower() == 'sgd':
            opt = SGD(lr=0.1, **optimizerParams)
        model = Model(layerIn, activationLayer)
        model.compile(loss=lossFct, optimizer=opt)
        
        self.model = model
        if self.verboseBuild:            
            model.summary()
            logging.debug(model.get_config())            
            logging.debug("Optimizer: %s, %s" % (str(type(opt)), str(opt.get_config())))

    def evaluate(self, epochs):  
        logging.info("%d train sentences" % len(self.dataset['trainMatrix']))     
        logging.info("%d dev sentences" % len(self.dataset['devMatrix']))   
        logging.info("%d test sentences" % len(self.dataset['testMatrix']))   
        
        devMatrix = self.dataset['devMatrix']
        testMatrix = self.dataset['testMatrix']
   
        total_train_time = 0
        no_improvement_since = 0
        
        for epoch in range(epochs):      
            sys.stdout.flush()           
            logging.info("--------- Epoch %d -----------" % (epoch+1))
            
            start_time = time.time()
            self.trainModel()
            time_diff = time.time() - start_time
            total_train_time += time_diff
            logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))
            
            
            start_time = time.time()
            dev_score, test_score = self.computeScores(devMatrix, testMatrix)
            
            if dev_score > self.max_dev_score:
                no_improvement_since = 0
                self.max_dev_score = dev_score 
                self.max_test_score = test_score
                
                if self.modelSavePath != None:
                    if self.devEqualTest:
                        savePath = self.modelSavePath.replace("[TestScore]_", "")
                    savePath = self.modelSavePath.replace("[DevScore]", "%.4f" % dev_score).replace("[TestScore]", "%.4f" % test_score).replace("[Epoch]", str(epoch))
                    directory = os.path.dirname(savePath)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        
                    if not os.path.isfile(savePath):
                        self.model.save(savePath, False)
                        import json
                        import h5py
                        mappingsJson = json.dumps(self.mappings)
                        with h5py.File(savePath, 'a') as h5file:
                            h5file.attrs['mappings'] = mappingsJson
                            h5file.attrs['maxCharLen'] = str(self.maxCharLen)
                            
                        #mappingsOut = open(savePath+'.mappings', 'wb')                        
                        #pkl.dump(self.dataset['mappings'], mappingsOut)
                        #mappingsOut.close()
                    else:
                        logging.info("Model", savePath, "already exists")
            else:
                no_improvement_since += 1
                
                
            if self.resultsOut != None:
                self.resultsOut.write("\t".join(map(str, [epoch+1, dev_score, test_score, self.max_dev_score, self.max_test_score])))
                self.resultsOut.write("\n")
                self.resultsOut.flush()
                
            logging.info("Max: %.4f on dev; %.4f on test" % (self.max_dev_score, self.max_test_score))
            logging.info("%.2f sec for evaluation" % (time.time() - start_time))
            
            if self.params['earlyStopping'] > 0 and no_improvement_since >= self.params['earlyStopping']:
                logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
                break
            
            
    def computeScores(self, devMatrix, testMatrix):
        return self.computeF1Scores(devMatrix, testMatrix)
            
    def computeF1Scores(self, devMatrix, testMatrix):
        logging.info("Dev-Data metrics:")
        dev_f1s = 0
        for tag in self.label2Idx.keys():
            dev_pre, dev_rec, dev_f1, dev_tags = self.computeF1(devMatrix, 'dev', self.label2Idx[tag])
            logging.info("[%s]: Prec: %.3f, Rec: %.3f, F1: %.4f" % (tag, dev_pre, dev_rec, dev_f1))
            dev_f1s += dev_f1

        dev_f1 = dev_f1s / float(len(self.label2Idx))
        test_f1 = dev_f1
        if not self.devEqualTest:
            logging.info("")
            logging.info("Test-Data metrics:")
            test_f1s = 0
            for tag in self.label2Idx.keys():
                test_pre, test_rec, test_f1 , test_tags = self.computeF1(testMatrix, 'test', self.label2Idx[tag])
                logging.info("[%s]: Prec: %.3f, Rec: %.3f, F1: %.4f" % (tag, test_pre, test_rec, test_f1))
                test_f1s += test_f1
            test_f1 = test_f1s / float(len(self.label2Idx))

        max_score = self.max_scores['dev']
        if self.writeOutput and max_score < dev_f1:
            self.writeOutputToFile(devMatrix, dev_tags, '%.4f_%s' % (dev_f1, 'dev'))
            self.max_scores['dev'] = dev_f1
        max_score = self.max_scores['test']
        if self.writeOutput and max_score < test_f1:
            self.writeOutputToFile(testMatrix, test_tags, '%.4f_%s' % (test_f1, 'test'))
            self.max_scores['test'] = test_f1
        return dev_f1, test_f1


    def tagSentences(self, sentences):
    
        paddedPredLabels = self.predictLabels(sentences)        
        predLabels = []
        for idx in range(len(sentences)):           
            unpaddedPredLabels = []
            for tokenIdx in range(len(sentences[idx]['tokens'])):
                if sentences[idx]['tokens'][tokenIdx] != 0: #Skip padding tokens                     
                    unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])

            predLabels.append(unpaddedPredLabels)

        idx2Label = {v: k for k, v in self.mappings['label'].items()}
        labels = [[idx2Label[tag] for tag in tagSentence] for tagSentence in predLabels]
        
        return labels
    
    def computeF1(self, sentences, name, tag_id):
        correctLabels = []
        predLabels = []
        paddedPredLabels = self.predictLabels(sentences)        
        
        for idx in range(len(sentences)):
            unpaddedCorrectLabels = []
            unpaddedPredLabels = []
            for tokenIdx in range(len(sentences[idx]['tokens'])):
                if sentences[idx]['tokens'][tokenIdx] != 0: #Skip padding tokens 
                    unpaddedCorrectLabels.append(sentences[idx][self.labelKey][tokenIdx])
                    unpaddedPredLabels.append(paddedPredLabels[idx][tokenIdx])
            correctLabels.append(unpaddedCorrectLabels)
            predLabels.append(unpaddedPredLabels)

        pre, rec, f1  =  compute_f1_token_basis(predLabels, correctLabels, tag_id)
        
        return pre, rec, f1, predLabels
    
    def writeOutputToFile(self, sentences, predLabels, name):
            outputName = 'tmp/'+name
            fOut = open(outputName, 'w')
            for sentenceIdx in range(len(sentences)):
                for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
                    token = self.idx2Word[sentences[sentenceIdx]['tokens'][tokenIdx]]
                    label = self.idx2Label[sentences[sentenceIdx][self.labelKey][tokenIdx]]
                    predLabel = self.idx2Label[predLabels[sentenceIdx][tokenIdx]]
                    fOut.write("\t".join([token, label, predLabel]))
                    fOut.write("\n")
                fOut.write("\n")
            fOut.close()
    
    def loadModel(self, modelPath):
        import h5py
        import json
        from neuralnets.keraslayers.ChainCRF import create_custom_objects
        
        model = keras.models.load_model(modelPath, custom_objects=create_custom_objects())

        with h5py.File(modelPath, 'r') as f:
            mappings = json.loads(f.attrs['mappings'])
                
            if 'maxCharLen' in f.attrs and f.attrs['maxCharLen'] != 'None':
                self.maxCharLen = int(f.attrs['maxCharLen'])
            
        self.model = model        
        self.setMappings(None, mappings)
