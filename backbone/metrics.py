import backbone as b
from config import *


def precision(tp, fp):
    info_file = Config().getInfoFile()
    p = tp/(tp + fp)
    b.myPrint("Precision: " + str(p), info_file)
    return p

def sensitivity(tp, fn):    # also called True Positive Rate
    info_file = Config().getInfoFile()
    s = tp/(tp + fn)
    b.myPrint("Sensitivity: " + str(s), info_file)
    return s

def FPR(fp, tn):            # False Positive Rate
    info_file = Config().getInfoFile()
    f = fp/(fp + tn)
    b.myPrint("False Positive Rate : " + str(f), info_file)
    return f

def F1_score(precision, sensitivity, beta=2):
    info_file = Config().getInfoFile()
    f1 = (1 + beta**2) * ((precision * sensitivity)/(beta**2 * precision + sensitivity))
    b.myPrint("F1-Score : " + str(f1), info_file)
    return f1
