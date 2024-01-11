import numpy as np
from scipy.optimize import minimize

# Montant total à placer
montant_total = 30000

# Informations sur les comptes
comptes = [
    {"plafond": 10000, "taux_interet": 0.06},
    {"plafond": 12000, "taux_interet": 0.03},
    {"plafond": 22950, "taux_interet": 0.03},
]

# Fonction objectif à maximiser
def objectif_placement(x, comptes):
    return -sum(compte["taux_interet"] * xi for compte, xi in zip(comptes, x))

# Contrainte de somme totale
contrainte_somme = lambda x: np.sum(x) - montant_total

# Initialisation de la répartition
x0 = [0.0] * len(comptes)

# Contraintes
contraintes = [{"type": "eq", "fun": contrainte_somme}]

# Bounds pour chaque compte
bounds = [(0, compte["plafond"]) for compte in comptes]

# Exécution de l'optimisation
resultats_optimisation = minimize(
    objectif_placement,
    x0,
    args=(comptes,),
    constraints=contraintes,
    bounds=bounds
)

# Afficher les résultats
repartition_optimale = resultats_optimisation.x
for compte, xi in zip(comptes, repartition_optimale):
    print(f"Montant à placer dans le compte : {round(xi, 2)} € (Taux d'intérêt : {compte['taux_interet']*100}%)")
