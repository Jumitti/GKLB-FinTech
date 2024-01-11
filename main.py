import json

import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
from scipy.optimize import minimize


# Chargement des comptes depuis le fichier JSON
def charger_comptes():
    try:
        with open("comptes.json", "r") as file:
            comptes = json.load(file)
    except FileNotFoundError:
        comptes = []
    return comptes


# Enregistrement des comptes dans le fichier JSON
def enregistrer_comptes(comptes):
    with open("comptes.json", "w") as file:
        json.dump(comptes, file, indent=2)


# Calcul des soldes prévisionnels
def calculer_soldes_previsionnels(comptes, duree_annees=20, annee_en_cours=2024):
    for compte in comptes:
        solde_previsionnel = compte["solde"]
        solde_previsionnel_list = [solde_previsionnel]

        for annee in range(1, duree_annees + 1):
            solde_annuel_calculable = min(solde_previsionnel, compte["plafond"])
            interet_annuel = solde_annuel_calculable * (compte["taux_interet"] / 100)
            solde_previsionnel += interet_annuel
            solde_previsionnel_list.append(solde_previsionnel)

        compte["solde_previsionnel"] = solde_previsionnel_list
        compte["annee_en_cours"] = annee_en_cours


def calculer_solde_total(comptes, duree_annees=20, annee_en_cours=2024):
    solde_total = [0] * (duree_annees + 1)

    for compte in comptes:
        solde_total = [solde + solde_compte for solde, solde_compte in zip(solde_total, compte["solde_previsionnel"])]

    return solde_total


# Fonction pour calculer les intérêts gagnés
def calculer_interets(comptes, duree_annees=20, annee_en_cours=2024):
    interets_par_compte = {compte["nom"]: [0] * (duree_annees + 1) for compte in comptes}

    for annee in range(1, duree_annees + 1):
        for compte in comptes:
            solde_annuel_calculable = min(compte["solde_previsionnel"][annee - 1], compte["plafond"])
            interet_annuel = solde_annuel_calculable * (compte["taux_interet"] / 100)
            interets_par_compte[compte["nom"]][annee] = interet_annuel

    return interets_par_compte


# Fonction pour calculer la somme totale des intérêts
def calculer_interets_total(interets_par_compte, duree_annees=20):
    interets_total = [0] * (duree_annees + 1)

    for compte, interets in interets_par_compte.items():
        for annee in range(duree_annees + 1):
            interets_total[annee] += interets[annee]

    return interets_total


# Fonction objectif à maximiser
def objectif_placement(x, comptes):
    return -sum(compte["taux_interet"] * xi for compte, xi in zip(comptes, x))


# Fonction principale de l'application
def main():
    st.title("Gestion des Comptes d'Épargne")

    # Charger les comptes depuis le fichier JSON
    comptes = charger_comptes()

    # Sidebar pour ajouter ou modifier un compte
    st.sidebar.header("Modifier un compte")

    # Sélectionner un compte existant
    compte_selectionne = st.sidebar.selectbox("Sélectionner un compte", [""] + [compte["nom"] for compte in comptes])

    # Si un compte est sélectionné, afficher ses détails et permettre la modification ou la suppression
    if compte_selectionne:
        compte_a_modifier = next((compte for compte in comptes if compte["nom"] == compte_selectionne), None)
        if compte_a_modifier:
            st.sidebar.write(f"**Compte Sélectionné : {compte_selectionne}**")
            solde_initial = st.sidebar.number_input("Solde initial", value=float(compte_a_modifier["solde"]), step=1.0)
            taux_interet = st.sidebar.number_input("Taux d'intérêt (%)", value=float(compte_a_modifier["taux_interet"]),
                                                   step=0.1)
            plafond = st.sidebar.number_input("Plafond du compte", value=float(compte_a_modifier["plafond"]), step=1.0)

            # Mettre à jour les informations du compte sélectionné
            compte_a_modifier["solde"] = solde_initial
            compte_a_modifier["taux_interet"] = taux_interet
            compte_a_modifier["plafond"] = plafond
            enregistrer_comptes(comptes)

            # Bouton pour supprimer le compte
            if st.sidebar.button("Supprimer le compte"):
                comptes.remove(compte_a_modifier)
                enregistrer_comptes(comptes)
                compte_selectionne = ""

    st.sidebar.header("Ajouter un compte")
    # Bouton pour ajouter un nouveau compte
    nouveau_compte_nom = st.sidebar.text_input("Nom du nouveau compte")
    solde_initial_nouveau = st.sidebar.number_input("Solde initial du nouveau compte", step=1.0)
    taux_interet_nouveau = st.sidebar.number_input("Taux d'intérêt (%) du nouveau compte", step=0.1)
    plafond_nouveau = st.sidebar.number_input("Plafond du nouveau compte", step=1.0)

    if any(compte["nom"] == nouveau_compte_nom for compte in comptes):
        st.sidebar.error("Le nom existe deja")
        disable = True
    else:
        disable = False

    if st.sidebar.button("Ajouter le nouveau compte", disabled=disable):
        nouveau_compte = {
            "nom": nouveau_compte_nom,
            "solde": solde_initial_nouveau,
            "taux_interet": taux_interet_nouveau,
            "plafond": plafond_nouveau
        }
        comptes.append(nouveau_compte)
        enregistrer_comptes(comptes)
        compte_selectionne = nouveau_compte_nom

    # Calculer les soldes prévisionnels sur 20 ans à partir de 2024 (modifiable)
    annee_en_cours = 2024
    calculer_soldes_previsionnels(comptes, duree_annees=20, annee_en_cours=annee_en_cours)

    # Calculer les intérêts gagnés
    interets_par_compte = calculer_interets(comptes, duree_annees=20, annee_en_cours=annee_en_cours)

    # Calculer le solde total et la somme des intérêts totaux
    solde_total = calculer_solde_total(comptes)
    interets_total = calculer_interets_total(interets_par_compte)

    # Créer un DataFrame pandas pour utiliser avec Altair
    df = pd.DataFrame()

    # Ajouter les colonnes pour le graphique prévisionnel
    df["Année"] = list(range(annee_en_cours, annee_en_cours + 21))
    df["Type"] = "Total"
    df["Montant"] = solde_total

    # Ajouter les colonnes pour chaque compte dans le graphique prévisionnel
    for compte in comptes:
        df_compte = pd.DataFrame({
            'Année': list(range(annee_en_cours, annee_en_cours + 21)),
            'Type': compte["nom"],
            'Montant': compte["solde_previsionnel"]
        })
        df = pd.concat([df, df_compte], ignore_index=True)

    # Afficher le graphique prévisionnel
    st.header("Graphique Prévisionnel avec Solde Total pour Chaque Compte")
    opacity = alt.selection_single(fields=['Type'], on='click', bind='legend')
    chart = alt.Chart(df).encode(
        x='Année',
        y='Montant',
        color='Type',
        tooltip=['Année', 'Type', 'Montant'], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ).properties(
        width=600,
        height=400
    ).mark_line().interactive() + alt.Chart(df).mark_circle(size=100).encode(
        x='Année',
        y='Montant',
        color='Type',
        tooltip=['Année', 'Type', 'Montant'], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ).add_params(opacity)

    st.altair_chart(chart, theme=None, use_container_width=True)

    # Créer un DataFrame pandas pour utiliser avec Altair
    df_interets = pd.DataFrame()

    # Ajouter les colonnes pour le graphique des intérêts
    df_interets["Année"] = list(range(annee_en_cours, annee_en_cours + 21))
    df_interets["Type"] = "Total"
    df_interets["Intérêts"] = interets_total

    # Ajouter les colonnes pour chaque compte dans le graphique des intérêts
    for compte in comptes:
        df_interets_compte = pd.DataFrame({
            'Année': list(range(annee_en_cours, annee_en_cours + 21)),
            'Type': compte["nom"],
            'Intérêts': interets_par_compte[compte["nom"]]
        })
        df_interets = pd.concat([df_interets, df_interets_compte], ignore_index=True)

    # Afficher le graphique des intérêts avec la somme totale des intérêts
    st.header("Graphique des Intérêts avec Intérêts Total pour Chaque Compte")
    opacity = alt.selection_single(fields=['Type'], on='click', bind='legend')
    chart_interets_total = alt.Chart(df_interets).encode(
        x='Année',
        y='Intérêts',
        color='Type',
        tooltip=['Année', 'Type', 'Intérêts'], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ).properties(
        width=600,
        height=400
    ).mark_line().interactive() + alt.Chart(df_interets).mark_circle(size=100).encode(
        x='Année',
        y='Intérêts',
        color='Type',
        tooltip=['Année', 'Type', 'Intérêts'], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ).add_params(opacity)

    st.altair_chart(chart_interets_total, theme=None, use_container_width=True)

    # Afficher la liste des comptes
    st.header("Liste des Comptes")
    for compte in comptes:
        st.write(
            f"**{compte['nom']}**: Solde actuel - {compte['solde']} €, Taux d'intérêt - {compte['taux_interet']}%, Plafond - {compte['plafond']} €")

    # Afficher la répartition optimale dans Streamlit
    st.header("Répartition Optimale des Fonds pour Maximiser les Intérêts")

    # Sélection des comptes dans Streamlit avec st.checkbox
    comptes_selectionnes = [st.checkbox(compte["nom"], value=True) for compte in comptes]

    # Filtrer les comptes en fonction de la sélection
    comptes_selectionnes = [compte for compte, selectionne in zip(comptes, comptes_selectionnes) if selectionne]

    # Montant total à placer
    montant_total = sum(compte["solde"] for compte in comptes_selectionnes)

    # Contraintes
    contrainte_somme = lambda x: np.sum(x) - montant_total

    contraintes = [{"type": "eq", "fun": contrainte_somme}]

    # Initialisation de la répartition
    x0 = [0.0] * len(comptes_selectionnes)

    # Bounds pour chaque compte
    bounds = [(0, compte["plafond"]) for compte in comptes_selectionnes]

    # Exécution de l'optimisation
    resultats_optimisation = minimize(
        objectif_placement,
        x0,
        args=(comptes_selectionnes,),
        constraints=contraintes,
        bounds=bounds
    )

    # Affichage des résultats dans Streamlit
    repartition_optimale = resultats_optimisation.x

    infos_json = []
    for compte, xi in zip(comptes_selectionnes, repartition_optimale):
        st.write(
            f"Montant à placer sur {compte['nom']} : {round(xi, 2)} € (Taux d'intérêt : {compte['taux_interet']}% | Plafond: {compte['plafond']})")
        info_compte = {
            "nom": compte["nom"],
            "solde": round(xi, 2),
            "taux_interet": compte["taux_interet"],
            "plafond": compte["plafond"]
        }
        infos_json.append(info_compte)

    json_str = json.dumps(infos_json, indent=2)
    comptes = json.loads(json_str)

    # Calculer les soldes prévisionnels sur 20 ans à partir de 2024 (modifiable)
    annee_en_cours = 2024
    calculer_soldes_previsionnels(comptes, duree_annees=20, annee_en_cours=annee_en_cours)

    # Calculer les intérêts gagnés
    interets_par_compte = calculer_interets(comptes, duree_annees=20, annee_en_cours=annee_en_cours)

    # Calculer le solde total et la somme des intérêts totaux
    solde_total = calculer_solde_total(comptes)
    interets_total = calculer_interets_total(interets_par_compte)

    # Créer un DataFrame pandas pour utiliser avec Altair
    df = pd.DataFrame()

    # Ajouter les colonnes pour le graphique prévisionnel
    df["Année"] = list(range(annee_en_cours, annee_en_cours + 21))
    df["Type"] = "Total"
    df["Montant"] = solde_total

    # Ajouter les colonnes pour chaque compte dans le graphique prévisionnel
    for compte in comptes:
        df_compte = pd.DataFrame({
            'Année': list(range(annee_en_cours, annee_en_cours + 21)),
            'Type': compte["nom"],
            'Montant': compte["solde_previsionnel"]
        })
        df = pd.concat([df, df_compte], ignore_index=True)

    # Afficher le graphique prévisionnel
    st.header("Graphique Prévisionnel avec Solde Total pour Chaque Compte")
    chart = alt.Chart(df).mark_line().encode(
        x='Année',
        y='Montant',
        color='Type',
        tooltip=['Année', 'Type', 'Montant']
    ).properties(
        width=600,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

    # Créer un DataFrame pandas pour utiliser avec Altair
    df_interets = pd.DataFrame()

    # Ajouter les colonnes pour le graphique des intérêts
    df_interets["Année"] = list(range(annee_en_cours, annee_en_cours + 21))
    df_interets["Type"] = "Total"
    df_interets["Intérêts"] = interets_total

    # Ajouter les colonnes pour chaque compte dans le graphique des intérêts
    for compte in comptes:
        df_interets_compte = pd.DataFrame({
            'Année': list(range(annee_en_cours, annee_en_cours + 21)),
            'Type': compte["nom"],
            'Intérêts': interets_par_compte[compte["nom"]]
        })
        df_interets = pd.concat([df_interets, df_interets_compte], ignore_index=True)

    # Afficher le graphique des intérêts avec la somme totale des intérêts
    st.header("Graphique des Intérêts avec Intérêts Total pour Chaque Compte")
    chart_interets_total = alt.Chart(df_interets).mark_line().encode(
        x='Année',
        y='Intérêts',
        color='Type',
        tooltip=['Année', 'Type', 'Intérêts']
    ).properties(
        width=600,
        height=400
    )

    st.altair_chart(chart_interets_total, use_container_width=True)

if __name__ == "__main__":
    main()
