import numpy as np
from sklearn.metrics import roc_curve

# EER: Schnittpunkt von FAR und FRR
# FAR = FPR (x-Achse der ROC-Kurve)
# FRR = 1 - TPR (nicht direkt auf ROC-Kurve, wird umgerechnet)
# EER = (FAR + FRR) / 2 am Punkt wo |FAR - FRR| minimal
# EXTRA INFO: FAR = FP/(FP+TN),  FRR = FN/(FN+TP) = 1 - TPR
# labels: 1 = gleicher Sprecher, 0 = verschiedene Sprecher
# scores: höher = ähnlicher
def eer_compute(fpr, tpr, thresholds):
    far = fpr # FAR = False Acceptance Rate
    frr = 1.0 - tpr # FRR = False Rejection Rate
    index = np.argmin(np.abs(far - frr)) # Index des Punktes mit minimalem Abstand zwischen FAR und FRR
    eer_wert = (fpr[index] + frr[index]) / 2.0
    threshold = float(thresholds[index]) # Schwellenwert am EER-Punkt, ab dem die Entscheidung von "unähnlich" zu "ähnlich" wechselt
    return round(eer_wert * 100, 3), round(threshold, 3)

# Minimum Detection Cost Function
# Basierend auf: NIST SRE 2016 Evaluation Plan, Section 3
# https://www.nist.gov/system/files/documents/2016/10/07/sre16_eval_plan_v1.3.pdf
# Adaptiert von Kaldi (Apache License 2.0):
# https://github.com/kaldi-asr/kaldi/blob/master/egs/sre08/v1/sid/compute_min_dcf.py
# fnrs:       False Negative Rates (FRR) für jeden Schwellenwert, berechnet aus 1 - TPR
# fprs:       False Positive Rates (FAR) für jeden Schwellenwert, direkt aus roc_curve
# p_target:   A-priori-Wahrscheinlichkeit, dass es der echte Sprecher ist (NIST-Standard: 0.01)
# c_miss:     Kosten für fälschliche Ablehnung eines echten Sprechers (NIST-Standard: 1), desto höher, desto schwerwiegender ist es, einen echten Sprecher abzulehnen
# c_fa:       Kosten für fälschliche Akzeptanz eines fremden Sprechers (NIST-Standard: 1), desto höher, desto schwerwiegender ist es, einen fremden Sprecher zu akzeptieren
def compute_min_dcf(fnrs, fprs, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return round(min_c_det / c_def, 3)

# TAR@FAR: True Accept Rate bei gegebener False Accept Rate
# FAR = Wahrscheinlichkeit, dass ein fremder Sprecher fälschlicherweise akzeptiert wird
def tar_at_far(target_far, fpr, tpr):
    return round(np.interp(target_far, fpr, tpr) * 100, 3)

def all_metrics(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnrs = 1.0 - tpr # FNR = False Negative Rate = FRR = False Rejection Rate, fnrs ist plural, weil Kaldi Implementierung von compute_min_dcf fnrs als Liste von FNRs für alle Schwellenwerte erwartet
    eer_value, threshold = eer_compute(fpr, tpr, thresholds)
    return {
        "eer_%":        eer_value,
        "eer_schwelle": threshold,
        "min_dcf":      compute_min_dcf(fnrs, fpr, 0.01, 1, 1),
        "tar@far=1%":   tar_at_far(0.01, fpr, tpr),
        "tar@far=5%":   tar_at_far(0.05, fpr, tpr),
    }
