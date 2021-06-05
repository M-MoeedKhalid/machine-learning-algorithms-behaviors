def Getvalues(cm):
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    Sensitivity = round(TP / (TP + FN), 2)
    Specificity = round(TN / (TN + FP), 2)
    PPV = round(TP / (TP + FP), 2)
    NPV = round(TN / (TN + FN), 2)
    return Sensitivity, Specificity, PPV, NPV


names = [
    'datasets/circles0.3.csv',
    'datasets/halfkernel.csv',
    'datasets/moons1.csv',
    'datasets/spiral1.csv',
    'datasets/twogaussians42.csv',
    'datasets/twogaussians33.csv'
]