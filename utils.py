# Contient les fonctions d'erreur, les fonctions des modèles
# Fonctions d'erreur - https://www.mdpi.com/1996-1944/14/15/4222

import pandas as pd
import numpy as np
from math import sqrt


def err_rmse(Ymod, Yexp):
    """Calcul de l'erreur RMSE (root mean square error).
    Ymod : valeurs prédites par le modèle
    Yexp : valeurs expérimentales
    """
    n = len(Ymod)
    return sqrt(1/n*np.sum(np.square(Yexp - Ymod)))

def err_rse(Ymod, Yexp):
    """Calcul de l'erreur RSE (relative standard error).
    Ymod : valeurs prédites par le modèle
    Yexp : valeurs expérimentales
    """
    Yexp_mean = np.mean(Yexp)
    return np.sum(np.square(Yexp - Ymod))/np.sum(np.square(Yexp - Yexp_mean))

def err_r2(Ymod, Yexp):
    """Calcul de l'erreur R² (correlation coefficient).
    Ymod : valeurs prédites par le modèle
    Yexp : valeurs expérimentales
    """
    return (np.corrcoef(Ymod, Yexp)[0, 1])**2
