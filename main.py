import json

import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
from scipy.optimize import minimize


# Chargement des comptes depuis le fichier JSON
def load_saving_accounts():
    try:
        with open("saving_accounts.json", "r") as file:
            saving_accounts = json.load(file)
    except FileNotFoundError:
        saving_accounts = []
    return saving_accounts


# Enregistrement des comptes dans le fichier JSON
def save_saving_account(comptes):
    with open("saving_accounts.json", "w") as file:
        json.dump(comptes, file, indent=2)


# Barplot savings
def barplot_savings(comptes):
    df = pd.DataFrame(comptes)

    df = df.rename(columns={'nom': 'Name saving'})
    df = df.rename(columns={'solde': 'Sold (€)'})
    df = df.rename(columns={'taux_interet': 'Interest rate (%)'})
    df = df.rename(columns={'plafond': 'Limit (€)'})

    # Ajouter une nouvelle entrée pour le total
    df = pd.concat([df, pd.DataFrame([{"Name saving": "Total", "Sold (€)": df['Sold (€)'].sum()}])])

    # Barplot avec Altair
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Name saving:N', axis=alt.Axis(title=None)),
        y=alt.Y('Sold (€):Q', axis=alt.Axis(title='Sold')),
        color='Name saving',  # Couleur de la barre de solde
        tooltip=['Name saving:N', 'Sold (€):Q', 'Limit (€):Q', 'Interest rate (%):Q'],
    ).properties(
        width=400,
        height=300
    ).interactive()

    # Ajouter une barre transparente pour le plafond
    transparent_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Name saving:N', axis=alt.Axis(title=None)),
        y=alt.Y('Limit (€):Q', axis=alt.Axis(title='Limit')),
        color='Name saving',  # Couleur transparente pour le plafond
        opacity=alt.value(0.5),  # Opacité de la barre transparente
        tooltip=['Name saving:N', 'Sold (€):Q', 'Limit (€):Q', 'Interest rate (%):Q']
    ).properties(
        width=400,
        height=300
    ).interactive()

    # Afficher le graphique avec Streamlit
    st.altair_chart(chart + transparent_chart, theme=None, use_container_width=True)


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


def graphique_total(df, solde_total, annee_en_cours, comptes_selectionnes):
    # Ajouter les colonnes pour le graphique prévisionnel
    df["Année"] = list(range(annee_en_cours, annee_en_cours + 21))
    df["Type"] = "Total"
    df["Montant"] = solde_total

    # Ajouter les colonnes pour chaque compte dans le graphique prévisionnel
    for compte in comptes_selectionnes:
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


def graphique_interet(df_interets, interets_par_compte, interets_total, annee_en_cours, comptes_selectionnes):
    # Ajouter les colonnes pour le graphique des intérêts
    df_interets["Année"] = list(range(annee_en_cours, annee_en_cours + 21))
    df_interets["Type"] = "Total"
    df_interets["Intérêts"] = interets_total

    # Ajouter les colonnes pour chaque compte dans le graphique des intérêts
    for compte in comptes_selectionnes:
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


# Fonction objectif à maximiser
def objectif_placement(x, comptes):
    return -sum(compte["taux_interet"] * xi for compte, xi in zip(comptes, x))


# Main page
def main():
    # Load from json
    comptes = load_saving_accounts()

    # Ajouter un nouveau compte
    st.sidebar.title(f"📈 💸 [GK!LB](https://www.youtube.com/watch?v=S4Ez-aDbAoA)")
    st.sidebar.divider()
    with st.sidebar.expander("Add a saving account", expanded=False):
        new_account_name = st.text_input("Name")
        solde_initial_nouveau = st.number_input("Solde initial du nouveau compte", step=1.0)
        taux_interet_nouveau = st.number_input("Taux d'intérêt (%) du nouveau compte", step=0.1)
        plafond_nouveau = st.number_input("Plafond du nouveau compte", step=1.0)

        if any(compte["nom"] == new_account_name for compte in comptes):
            disable = True
        else:
            disable = False

        if st.button("Ajouter le nouveau compte", disabled=disable,
                             help="Le nom existe déjà" if disable else ""):
            nouveau_compte = {
                "nom": new_account_name,
                "solde": solde_initial_nouveau,
                "taux_interet": taux_interet_nouveau,
                "plafond": plafond_nouveau
            }
            comptes.append(nouveau_compte)
            save_saving_account(comptes)
            compte_selectionne = ""

    # Modifier un compte
    with st.sidebar.expander("Modifier un compte", expanded=False):
        compte_selectionne = st.selectbox("Sélectionner un compte", [""] + [compte["nom"] for compte in comptes],
                                                  index=0 if len(comptes) == 0 else 1)
        if compte_selectionne:
            compte_a_modifier = next((compte for compte in comptes if compte["nom"] == compte_selectionne), None)
            if compte_a_modifier:
                st.write(f"**Compte Sélectionné : {compte_selectionne}**")
                solde_initial = st.number_input("Solde initial", value=float(compte_a_modifier["solde"]), step=1.0)
                taux_interet = st.number_input("Taux d'intérêt (%)", value=float(compte_a_modifier["taux_interet"]),
                                                       step=0.1)
                plafond = st.number_input("Plafond du compte", value=float(compte_a_modifier["plafond"]), step=1.0)

                compte_a_modifier["solde"] = solde_initial
                compte_a_modifier["taux_interet"] = taux_interet
                compte_a_modifier["plafond"] = plafond
                save_saving_account(comptes)

                # Bouton pour supprimer le compte
                if st.button("Supprimer le compte"):
                    comptes.remove(compte_a_modifier)
                    save_saving_account(comptes)
                    compte_selectionne = ""

    st.sidebar.divider()

    # Actualiser
    if st.sidebar.button("Actualiser"):
        st.rerun()

    # Barplot savings
    barplot_savings(comptes)

    comptes = [{**compte, "is_widget": True} for compte in comptes]
    df = pd.DataFrame(comptes)
    edited_df = st.data_editor(df,
                               column_config={
                                   "nom": "Name saving",
                                   "solde": "Sold (€)",
                                   "taux_interet": "Interest rate (%)",
                                   "plafond": "Saving limit (€)",
                                   "is_widget": "Selection",
                               },
                               disabled=["nom", "solde", "taux_interet", "plafond"],
                               hide_index=True)

    # comptes_selected = edited_df.loc[edited_df["is_widget"], "nom"].tolist()
    comptes_selectionnes = [compte for compte in comptes if compte["nom"] in edited_df.loc[edited_df["is_widget"], "nom"].tolist()]

    # Montant total à placer
    montant_total = sum(compte["solde"] for compte in comptes_selectionnes)

    # Calculer les soldes prévisionnels sur 20 ans à partir de 2024 (modifiable)
    annee_en_cours = 2024
    calculer_soldes_previsionnels(comptes_selectionnes, duree_annees=20, annee_en_cours=annee_en_cours)

    # Calculer les intérêts gagnés
    interets_par_compte = calculer_interets(comptes_selectionnes, duree_annees=20, annee_en_cours=annee_en_cours)

    # Calculer le solde total et la somme des intérêts totaux
    solde_total = calculer_solde_total(comptes_selectionnes)
    interets_total = calculer_interets_total(interets_par_compte)

    # Créer un DataFrame pandas pour utiliser avec Altair
    df = pd.DataFrame()

    graphique_total(df, solde_total, annee_en_cours, comptes_selectionnes)

    # Créer un DataFrame pandas pour utiliser avec Altair
    df_interets = pd.DataFrame()

    graphique_interet(df_interets, interets_par_compte, interets_total, annee_en_cours, comptes_selectionnes)

    # Afficher la répartition optimale dans Streamlit
    st.header("Répartition Optimale des Fonds pour Maximiser les Intérêts")

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
    calculer_soldes_previsionnels(comptes, duree_annees=20, annee_en_cours=annee_en_cours)

    # Calculer les intérêts gagnés
    interets_par_compte = calculer_interets(comptes, duree_annees=20, annee_en_cours=annee_en_cours)

    # Calculer le solde total et la somme des intérêts totaux
    solde_total = calculer_solde_total(comptes)
    interets_total = calculer_interets_total(interets_par_compte)

    # Créer un DataFrame pandas pour utiliser avec Altair
    df = pd.DataFrame()

    graphique_total(df, solde_total, annee_en_cours, comptes)

    # Créer un DataFrame pandas pour utiliser avec Altair
    df_interets = pd.DataFrame()

    graphique_interet(df_interets, interets_par_compte, interets_total, annee_en_cours, comptes)


if __name__ == "__main__":
    main()
