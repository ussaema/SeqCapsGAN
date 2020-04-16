import pickle as pickle
import os
import sys
sys.path.append('../coco-caption')
from pycocoevalcap.pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.pycocoevalcap.cider.cider import Cider
from pycocoevalcap.pycocoevalcap.meteor.meteor import Meteor

def score(ref, hypo, scores):
    bleu = [s for s in scores if "Bleu" in s]
    rouge = [s for s in scores if "ROUGE" in s]
    cider = [s for s in scores if "CIDEr" in s]
    scorers = []
    if len(bleu):
        scorers.append((Bleu(len(bleu)), bleu))
    if len(rouge):
        scorers.append((Rouge(),"ROUGE_L"))
    if len(cider):
        scorers.append((Cider(),"CIDEr"))

    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores
    

def evaluate(cand, ref, scores):
    
    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
    
    # compute bleu score
    final_scores = score(ref, hypo, scores)
    
    return final_scores
    
   
    
    
    
    
    
    
    
    
    
    


