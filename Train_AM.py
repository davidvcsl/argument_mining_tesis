from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
from optparse import OptionParser

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

embeddingsPathOpt = {'levy':'levy_deps.words',
                     'glove':'glove.6B.300d.txt',
                     'glove100':'glove.6B.100d.txt',
                     'word2vec':'',
                    }
op = OptionParser(usage='Usage: python Train_AM [dataset] [embedding] [opts]')
op.add_option("--optimizer",
              dest="optimizer",
              default="nadam",
              help="nadam / adam / rmsprop / adadelta / adagrad / sgd")
op.add_option("--eval",
              dest="evalTest",
              default=True,
              help="If test.txt is evaluated in each epoch")
(opts, args) = op.parse_args()
if len(sys.argv) < 3:
    print(args)
    print("Usage: python Train_AM [dataset] [embedding] [opts]")
    exit()

######################################################
#
# Data preprocessing
#
######################################################


# :: Train / Dev / Test-Files ::

datasetName = sys.argv[1]
dataColumns = {1:'tokens', 2:'AM_TAG'}
labelKey = 'AM_TAG'
embeddingsPath = embeddingsPathOpt[sys.argv[2]]

#Parameters of the network
params = {'dropout': [0.25, 0.25],
          'LSTM-Size': [100],
          'optimizer': opts.optimizer,
          'miniBatchSize': 32}


frequencyThresholdUnknownTokens = 50 #If a token that is not in the pre-trained embeddings file appears at least 50 times in the train.txt, then a new embedding is generated for this word

datasetFiles = [
        (datasetName, dataColumns),
    ]


# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasetFiles)




######################################################
#
# The training of the network starts here
#
######################################################

#Load the embeddings and the dataset
embeddings, word2Idx, datasets = loadDatasetPickle(pickleFile)
data = datasets[datasetName]


print("Dataset:", datasetName)
print("Label key: ", labelKey)
print("Train Sentences:", len(data['trainMatrix']))
print("Dev Sentences:", len(data['devMatrix']))
print("Test Sentences:", len(data['testMatrix']))


model = BiLSTM(devEqualTest=not opts.evalTest,params=params)
model.setMappings(embeddings, data['mappings'])
model.setTrainDataset(data, labelKey)
model.verboseBuild = True
model.modelSavePath = "models/%s/%s/[DevScore]_[TestScore]_[Epoch].h5" % (datasetName, labelKey) #Enable this line to save the model to the disk
model.evaluate(100)
