import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

def compute_confusion(true_labels: list[int], distances: list[float], threshold: float) -> tuple[int, int, int, int]:
    """
     Berechnet TP, TN, FP, FN anhand von Labels, Distanzen und einem Threshold.

     :param true_labels: Wahre Labels (1 = gleicher Sprecher, 0 = Fremder)
     :param distances: Cosine-Abstände zwischen Embeddings
     :param threshold: Grenzwert -> (Distanz <= threshold) -> gleicher Sprecher
     :return: (TP, TN, FP, FN)
     """

    predicted = []
    for dist in distances:
        predicted.append(1 if dist <= threshold else 0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted).ravel().tolist()
    return tp, tn, fp, fn


def find_best_threshold(distances: list[float], true_labels: list[int]) -> dict:
    # roc_curve erwartet: höherer Score = positiver. Cosine-Distanz ist umgekehrt (kleiner = ähnlicher),
    # daher Negation. Sonst liegt die ROC-Kurve unter der theoretischen Diagonalen. Ohne Minus findet man den Threshold inf.
    # roc_curve probiert jeden einzigartigen Wert in neg_distances als Threshold aus und berechnet entsprechend TPR/FPR.
    neg_distances = [-d for d in distances]
    fpr, tpr, thresholds = roc_curve(true_labels, neg_distances)

    # Youden-Index: argmax(TPR - FPR)
    # Maximiert TPR - FPR. Also maximiert die Differenz zwischen der Rate der korrekt akzeptierten echten Sprecher (TPR) und der Rate der fälschlicherweise akzeptierten fremden Sprecher (FPR).
    # https://gist.github.com/twolodzko/4fae2980a1f15f8682d243808e5859bb
    idx = np.argmax(tpr - fpr)
    return {
        "threshold": round(float(-thresholds[idx]), 3), # Umwandeln in Cosine-Distanz, da roc_curve hier mit negierten Distanzen arbeitet
        "tpr_%": round(float(tpr[idx]) * 100, 2),
        "fpr_%": round(float(fpr[idx]) * 100, 2),
    }
