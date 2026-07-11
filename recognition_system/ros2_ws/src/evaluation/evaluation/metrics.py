import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

def compute_confusion(true_labels: list[int], distances: list[float], threshold: float) -> tuple[int, int, int, int]:
    """
    Berechnet TP, TN, FP, FN anhand von Labels, Distanzen und einem Threshold. Gibt (TP, TN, FP, FN) zurück.
    true_labels: wahre Labels (1 = gleicher Sprecher, 0 = fremder Sprecher). distances: Cosine-Distanzen (nicht
    Similarity) zwischen den Embeddings, Wertebereich: 0 identisch, 1 orthogonal und 2 entgegengesetzt.
    threshold: Grenzwert, ab dem eine Distanz als "gleicher Sprecher" gilt (if distance <= threshold = 1 else 0).
    """

    predicted = []
    for dist in distances:
        predicted.append(1 if dist <= threshold else 0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted).ravel().tolist()
    return tp, tn, fp, fn


def find_best_threshold(distances: list[float], true_labels: list[int]) -> dict:
    """
    Bestimmt per Youden-Index den Distanz-Schwellenwert, der TPR - FPR maximiert.
    roc_curve erwartet: höherer Score = positiver. Cosine-Distanz ist umgekehr (kleiner = ähnlicher, es erwartet somit Kosinus-Ähnlichkeit), daher werden
    die Distanzen negiert, bevor sie an roc_curve übergeben werden. Ohne die Negation läge die ROC-Kurve unter der
    theoretischen Diagonalen. roc_curve setzt den ersten Threshold immer künstlich auf inf. Ohne Negation gewinnt genau dieser
    Punkt. roc_curve probiert dabei jeden einzigartigen Wert in den negierten Distanzen als Threshold aus und berechnet
    jeweils TPR/FPR dazu.
    Der Youden-Index (argmax(TPR - FPR)) maximiert die Differenz zwischen der Rate der korrekt akzeptierten echten
    Sprecher (TPR) und der Rate der fälschlicherweise akzeptierten fremden Sprecher (FPR).
    Referenz Implementierung Youden-Index: https://gist.github.com/twolodzko/4fae2980a1f15f8682d243808e5859bb
    """
    neg_distances = [-d for d in distances]
    fpr, tpr, thresholds = roc_curve(true_labels, neg_distances)
    idx = np.argmax(tpr - fpr)
    return {
        "threshold": round(float(-thresholds[idx]), 3), # Umwandeln in Cosine-Distanz, da roc_curve hier mit negierten Distanzen arbeitet
        "tpr_%": round(float(tpr[idx]) * 100, 2),
        "fpr_%": round(float(fpr[idx]) * 100, 2),
    }
