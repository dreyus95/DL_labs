from sklearn.metrics import confusion_matrix
import numpy as np


def twoway_confusion_matrix(cm, index):
    binary_cm = [[0 for x in range(2)] for y in range(2)]
    for i in range(0, len(cm[0])):
        for j in range(0, len(cm[0])):
            if i == j == index:
                binary_cm[0][0] = cm[i][j]  # TP goes to 0,0
            elif i != j:
                if j == index:
                    binary_cm[0][1] += cm[i][j]  # accumulate FP
                elif i == index:
                    binary_cm[1][0] += cm[i][j]  # accumulate FN

    binary_cm[1][1] = np.sum(cm) - np.sum(binary_cm)  # TN = N - (TP + FP + FN)
    return binary_cm


def calculate(cm_all, m, classIndex):
    if m == "macro":
        cm = twoway_confusion_matrix(cm_all, classIndex)
        return cm[1][1], cm[0][0], cm[0][1], cm[1][0]
    elif m == "micro":
        TP = TN = FP = FN = 0
        cm = twoway_confusion_matrix(cm_all, classIndex)
        for i in range(0, len(cm[0])):
            TP += cm[1][1]
            TN += cm[0][0]
            FP += cm[0][1]
            FN += cm[1][0]
        return TP, TN, FP, FN


def accuracy(y_true, y_pred, averaging="micro"):
    cm_all = confusion_matrix(y_true, y_pred)

    if averaging == "micro":
        TPs = []
        TNs = []
        sum = 0.0
        for index in range(len(cm_all[0])):
            TP, TN, FP, FN = calculate(cm_all, "micro", index)
            TPs.append(TP)
            TNs.append(TN)
            sum += (TP + FP + TN + FN)
        return (1.0 * np.sum(TPs) + np.sum(TNs)) / sum
    if averaging == "macro":
        acc = 0
        for i in range(0, len(cm_all[0])):
            TP, TN, FP, FN = calculate(cm_all, "macro", i)
            acc += 1.0 * (TP + TN) / (TP + TN + FP + FN)
        return acc / len(cm_all[0])


def precision(y_true, y_pred, averaging="macro"):
    cm_all = confusion_matrix(y_true, y_pred)

    if averaging == "micro":
        TPs = []
        FPs = []
        for index in range(len(cm_all[0])):
            TP, TN, FP, FN = calculate(cm_all, "micro", index)
            TPs.append(TP)
            FPs.append(FP)
        return (1.0 * np.sum(TPs)) / (np.sum(TPs) + np.sum(FPs))

    if averaging == "macro":
        p = 0
        for i in range(0, len(cm_all[0])):
            TP, TN, FP, FN = calculate(cm_all, "macro", i)
            if TP + FP != 0:
                p += 1.0 * TP / (TP + FP)
        return p / len(cm_all[0])


def recall(y_true, y_pred, averaging="macro"):
    cm_all = confusion_matrix(y_true, y_pred)

    if averaging == "micro":
        TPs = []
        FNs = []
        for index in range(len(cm_all[0])):
            TP, TN, FP, FN = calculate(cm_all, "micro", index)
            TPs.append(TP)
            FNs.append(FN)
        return (1.0 * np.sum(TPs)) / (np.sum(TPs) + np.sum(FNs))
    elif averaging == "macro":
        p = 0
        for i in range(0, len(cm_all[0])):
            TP, TN, FP, FN = calculate(cm_all, "macro", i)
            if TP + FN != 0:
                p += 1.0 * TP / (TP + FN)
        return p / len(cm_all[0])


def f1(y_true, y_pred, averaging="macro"):
    P = precision(y_true, y_pred, averaging)
    R = recall(y_true, y_pred, averaging)
    return (2 * P * R) / (P + R)
