import sys

if len(sys.argv) < 4:
    print("Usage: python Eval.py taggedFile predictedFile reportPath")
    exit()

taggedFile = sys.argv[1]
predictedFile = sys.argv[2]
reportPath = sys.argv[3]

with open(taggedFile, 'r') as f:
    taggedText = f.read().split("\n")

with open(predictedFile, 'r') as f:
    predictedText = f.read().split("\n")

currentLine = 0
errors = []

while(currentLine < len(taggedText)-1): #Saca ultimo (\n)
    pWords = predictedText[currentLine].split("\t")
    tWords = taggedText[currentLine].split("\t")
    if len(pWords)<2: #Salto de linea en predicted
        del predictedText[currentLine]
        pWords = predictedText[currentLine].split("\t")
    word = pWords[0]
    pred = pWords[1]
    tagged = tWords[4]
    if word != tWords[1]:
        print("FIXLINE {}: tagged = {}, pred = {}".format(currentLine, tWords[1], word))
    if pred != tagged:
        errors.append([currentLine+1, word, pred, tagged])
    currentLine += 1

with open(reportPath, 'w') as save:
    acc = "%.2f" % (1 - len(errors)/float(len(taggedText)-1))
    save.write("Tag errors found: {} ({}% acc)\n\nErrors:\n".format(len(errors),acc))
    for info in errors:
        save.write("Line: {},\t Word: {}, Tag: {}, Pred: {}\n".format(info[0], info[1], info[3], info[2]))
save.close()
