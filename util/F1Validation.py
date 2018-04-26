import logging


def compute_f1_token_basis(predictions, correct, O_Label): 
       
    prec = compute_precision_token_basis(predictions, correct, O_Label)
    rec = compute_precision_token_basis(correct, predictions, O_Label)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1

def compute_precision_token_basis(guessed_sentences, correct_sentences, O_Label):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert(len(guessed) == len(correct))
        for idx in range(len(guessed)):
            
            if guessed[idx] == O_Label:
                count += 1
               
                if guessed[idx] == correct[idx]:
                    correctCount += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision
