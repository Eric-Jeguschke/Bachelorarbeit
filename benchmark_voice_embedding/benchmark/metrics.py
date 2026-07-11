import numpy as np
from sklearn.metrics import roc_curve
from numpy.typing import NDArray

def eer_compute(far: NDArray, tpr: NDArray) -> float:
    """
    Bestimmt die Equal Error Rate (EER) aus einer ROC-Kurve.
    EER: Schnittpunkt von FAR und FRR. FAR = FPR aus roc_curve (x-Achse der ROC-Kurve), TPR = y-Achse der ROC-Kurve.
    FRR = 1 - TPR wird daraus berechnet, steht nicht direkt auf der Kurve.
    EER = (FAR + FRR) / 2 am Punkt wo |FAR - FRR| minimal.
    EXTRA INFO: FAR = FP/(FP+TN), FRR = FN/(FN+TP) = 1 - TPR.
    labels: 1 = gleicher Sprecher, 0 = verschiedene Sprecher
    scores: höher = ähnlicher.
    """
    frr = 1.0 - tpr # FRR = False Rejection Rate. frr ist hier ein NDArray
    index = np.argmin(np.abs(far - frr)) # Index des Punktes mit minimalem Abstand zwischen FAR und FRR
    eer_wert = (far[index] + frr[index]) / 2.0
    return round(eer_wert * 100, 3)

def compute_min_dcf(fnrs: NDArray, fars: NDArray, p_target: float, c_miss: float, c_fa: float) -> float:
    """
    Berechnet die normierte Minimum Detection Cost Function (minDCF).
    Basierend auf: NIST SRE 2016 Evaluation Plan, Section 3
    (https://www.nist.gov/system/files/documents/2016/10/07/sre16_eval_plan_v1.3.pdf),
    Code adaptiert von Kaldi (Apache License 2.0, https://github.com/kaldi-asr/kaldi/blob/master/egs/sre08/v1/sid/compute_min_dcf.py).
    fnrs: False Negative Rates (FRR) je Schwellenwert, berechnet aus 1 - TPR.
    fars: False Accept Rates (FAR) je Schwellenwert
    p_target: A-priori-Wahrscheinlichkeit, dass es der echte Sprecher ist (NIST-Standard: 0.01).
    c_miss: Kosten für fälschliche Ablehnung eines echten Sprechers (NIST-Standard: 1), desto höher, desto schwerwiegender ist es, einen echten Sprecher abzulehnen
    c_fa: Kosten für fälschliche Akzeptanz eines fremden Sprechers (NIST-Standard: 1), desto höher, desto schwerwiegender ist es, einen fremden Sprecher zu akzeptieren
    """
    min_c_det = float("inf")
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fars[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return round(min_c_det / c_def, 3)

def tar_at_far(target_far: float, far: NDArray, tpr: NDArray) -> float:
    """
    Berechnet die True-Accept-Rate bei einer festen False-Accept-Rate.
    TAR@FAR: True Accept Rate bei gegebener False Accept Rate.
    FAR = Wahrscheinlichkeit, dass ein fremder Sprecher fälschlicherweise akzeptiert wird.
    Interpolation von TPR an einem festen FAR-Punkt (eigentlich FPR-Punkt, da roc_curve das fpr nennt) der ROC-Kurve. Code ist angelehnt an:
    scikit-learn-Beispiel: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    return round(np.interp(target_far, far, tpr) * 100, 3) # target_far und far sind gleich skaliert, deshalb direkt vergleichbar

def all_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    """
    Berechnet alle Bewertungsmetriken (EER, minDCF, TAR@FAR) aus Labels und Ähnlichkeits-Scores.
    scores: erwartet Cosinus-Ähnlichkeit (höher = ähnlicher)
    """
    far, tpr, _ = roc_curve(labels, scores, pos_label=1) # roc_curve nennt far eigentlich fpr (far = fpr), hier durchgehend als far bezeichnet.
    fnrs = 1.0 - tpr # FNR = False Negative Rate = FRR = False Rejection Rate, fnrs ist plural, weil Kaldi Implementierung von compute_min_dcf fnrs als Liste von FNRs für alle Schwellenwerte erwartet
    eer_value = eer_compute(far, tpr)
    return {
        "eer_%":        eer_value,
        "min_dcf":      compute_min_dcf(fnrs, far, 0.01, 1, 1),
        "tar@far=1%":   tar_at_far(0.01, far, tpr),
        "tar@far=5%":   tar_at_far(0.05, far, tpr),
    }
