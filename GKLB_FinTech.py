import json

import altair as alt
import pandas as pd
import streamlit as st
import numpy as np
from scipy.optimize import minimize
import os
from datetime import datetime


# Depositary folder
def get_depositary_name(file):
    with open(os.path.join(directory_path, file), 'r') as json_file:
        data = json.load(json_file)
        return data[0].get("depositary", "")


# Loading depositary information
def load_saving_accounts(file):
    try:
        with open(f"depositary/{file}", "r") as file:
            saving_accounts = json.load(file)
    except FileNotFoundError:
        saving_accounts = []
    return saving_accounts


# Update/create depositary information
def save_saving_account(depositary_file, comptes):
    with open(f"depositary/{depositary_file}", "w") as file:
        json.dump(comptes, file, indent=2)


# Barplot savings
def barplot_savings(comptes):
    df = pd.DataFrame(comptes)

    df = df.rename(columns={'nom': 'Name saving'})
    df = df.rename(columns={'solde': 'Sold (€)'})
    df = df.rename(columns={'taux_interet': 'Interest rate (%)'})
    df = df.rename(columns={'plafond': 'Limit (€)'})
    df = pd.concat([df, pd.DataFrame([{"Name saving": "Total", "Sold (€)": df['Sold (€)'].sum()}])])  # Total of savings

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Name saving:N', axis=alt.Axis(title=None)),
        y=alt.Y('Sold (€):Q', axis=alt.Axis(title='Sold')),
        color='Name saving',
        tooltip=['Name saving:N', 'Sold (€):Q', 'Limit (€):Q', 'Interest rate (%):Q'],
    ).properties(
        width=400,
        height=300
    ).interactive()

    transparent_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Name saving:N', axis=alt.Axis(title=None)),
        y=alt.Y('Limit (€):Q', axis=alt.Axis(title='Limit')),
        color='Name saving',
        opacity=alt.value(0.5),
        tooltip=['Name saving:N', 'Sold (€):Q', 'Limit (€):Q', 'Interest rate (%):Q']
    ).properties(
        width=400,
        height=300
    ).interactive()

    st.altair_chart(chart + transparent_chart, theme=None, use_container_width=True)


# Savings evolution
def calculer_soldes_previsionnels(comptes, duree_annees, annee_en_cours=2024):
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


# Total of savings evolution
def calculer_solde_total(comptes, duree_annees, annee_en_cours=2024):
    solde_total = [0] * (duree_annees + 1)

    for compte in comptes:
        solde_total = [solde + solde_compte for solde, solde_compte in zip(solde_total, compte["solde_previsionnel"])]

    return solde_total


# Graphic of savings
def graphique_total(df, solde_total, forcast, annee_en_cours, comptes_selectionnes, title):
    df["Année"] = list(range(annee_en_cours, annee_en_cours + forcast + 1))
    df["Type"] = "Total"
    df["Montant"] = solde_total

    for compte in comptes_selectionnes:
        df_compte = pd.DataFrame({
            'Année': list(range(annee_en_cours, annee_en_cours + forcast + 1)),
            'Type': compte["nom"],
            'Montant': compte["solde_previsionnel"]
        })
        df = pd.concat([df, df_compte], ignore_index=True)

    st.subheader(title)
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


# Interest rate evolution
def calculer_interets(comptes, duree_annees, annee_en_cours=2024):
    interets_par_compte = {compte["nom"]: [0] * (duree_annees + 1) for compte in comptes}

    for annee in range(1, duree_annees + 1):
        for compte in comptes:
            solde_annuel_calculable = min(compte["solde_previsionnel"][annee - 1], compte["plafond"])
            interet_annuel = solde_annuel_calculable * (compte["taux_interet"] / 100)
            interets_par_compte[compte["nom"]][annee] = interet_annuel

    return interets_par_compte


# Total of interest rate evolution
def calculer_interets_total(interets_par_compte, duree_annees):
    interets_total = [0] * (duree_annees + 1)

    for compte, interets in interets_par_compte.items():
        for annee in range(duree_annees + 1):
            interets_total[annee] += interets[annee]

    return interets_total


# Graphic of interest rate
def graphique_interet(df_interets, interets_par_compte, forcast, interets_total, annee_en_cours, comptes_selectionnes, title):
    df_interets["Année"] = list(range(annee_en_cours, annee_en_cours + forcast + 1))
    df_interets["Type"] = "Total"
    df_interets["Intérêts"] = interets_total

    for compte in comptes_selectionnes:
        df_interets_compte = pd.DataFrame({
            'Année': list(range(annee_en_cours, annee_en_cours + forcast + 1)),
            'Type': compte["nom"],
            'Intérêts': interets_par_compte[compte["nom"]]
        })
        df_interets = pd.concat([df_interets, df_interets_compte], ignore_index=True)

    st.subheader(title)
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


# Maximizing investments
def objectif_placement(x, comptes):
    return -sum(compte["taux_interet"] * xi for compte, xi in zip(comptes, x))

# Contrainte de somme totale
def contrainte_somme(x, montant_total):
    return sum(x) - montant_total


# Settings for Streamlit page
st.set_page_config(
    page_title="GK!LB",
    page_icon="💸",
    layout="wide")

# Main page
st.title('📈 💸 GK!LB - FinTech')
st.divider()

# Sidebar manager
st.sidebar.title(f"[GK!LB - Manager](https://www.youtube.com/watch?v=S4Ez-aDbAoA)")

if st.sidebar.button("🔄️ Update"):  # Update info
    st.toast("Updated", icon="🔄️")
st.sidebar.divider()

# Load all depositary
directory_path = "depositary"
json_files = [file for file in os.listdir(directory_path) if file.endswith(".json")]

if len(json_files) > 0:
    json_mapping = {}
    for file in json_files:
        file_path = os.path.join(directory_path, file)

        try:
            with open(file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                if isinstance(data, list) and data and isinstance(data[0], dict) and "depositary" in data[0]:
                    depositary = data[0]["depositary"]
                    json_mapping[depositary] = file
                else:
                    st.error(f"File {file} does not have the expected structure.")
        except Exception as e:
            st.error(f"Error reading file {file}: {e}")

    # Depositary selection
    selected_json_file = st.sidebar.selectbox("Sélectionnez le dépositaire :", list(json_mapping.keys()))
    depositary_file = json_mapping[selected_json_file]
    comptes = load_saving_accounts(depositary_file)

    # Add a saving account
    with st.sidebar.expander("Add a saving account", expanded=False):
        new_account_name = st.text_input("Name")
        solde_initial_nouveau = st.number_input("Solde initial du nouveau compte", step=1.0)
        taux_interet_nouveau = st.number_input("Taux d'intérêt (%) du nouveau compte", step=0.1)
        plafond_nouveau = st.number_input("Plafond du nouveau compte", step=1.0)

        if any(isinstance(compte, dict) and "nom" in compte and compte["nom"] == new_account_name for compte in
               comptes):
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
            save_saving_account(depositary_file, comptes)
            st.toast("Nouveau compte")

    # Update a saving account
    with st.sidebar.expander("Modifier un compte", expanded=False):
        compte_selectionne = st.selectbox(
            "Sélectionner un compte",
            [compte["nom"] for compte in comptes if isinstance(compte, dict) and "nom" in compte],
            index=0)

        if compte_selectionne:
            compte_a_modifier = next((compte for compte in comptes if
                                      isinstance(compte, dict) and "nom" in compte and compte[
                                          "nom"] == compte_selectionne), None)
            if compte_a_modifier:
                solde_initial = st.number_input("Solde initial", value=float(compte_a_modifier["solde"]), step=1.0)
                taux_interet = st.number_input("Taux d'intérêt (%)", value=float(compte_a_modifier["taux_interet"]),
                                               step=0.1)
                plafond = st.number_input("Plafond du compte", value=float(compte_a_modifier["plafond"]), step=1.0)
                compte_a_modifier["solde"] = solde_initial
                compte_a_modifier["taux_interet"] = taux_interet
                compte_a_modifier["plafond"] = plafond
                save_saving_account(depositary_file, comptes)

                # Delete saving account
                if st.button("Supprimer le compte"):
                    comptes.remove(compte_a_modifier)
                    save_saving_account(depositary_file, comptes)
                    st.toast("Compte modifié")

    # Delete a depositary
    with st.sidebar.expander("Supprimer le dépositaire"):
        st.warning(f"Êtes-vous sûr de vouloir supprimer le dépositaire '{selected_json_file}'?")
        if st.button("Oui"):
            os.remove(os.path.join(directory_path, depositary_file))
            st.toast(f"Le dépositaire '{selected_json_file}' a été supprimé avec succès.")

    st.sidebar.divider()

# Create a depositary
with st.sidebar.expander("Ajouter un dépositaire", expanded=False):
    depositary_name = st.text_input("Nom du dépositaire")
    new_account_name = st.text_input("Nom du nouveau compte")
    solde_initial_nouveau = st.number_input("Solde initial du nouveau compte", step=1.0, key='1')
    taux_interet_nouveau = st.number_input("Taux d'intérêt (%) du nouveau compte", step=0.1, key='2')
    plafond_nouveau = st.number_input("Plafond du nouveau compte", step=1.0, key='3')

    depositary_file = f"depositary_{depositary_name}.json"
    if len(json_files) > 0:
        if os.path.exists(f'depositary/{depositary_file}'):
            disable = True
        else:
            disable = False
    else:
        disable = False

    if st.button("Ajouter le nouveau compte", disabled=disable,
                 help="Le nom existe déjà" if disable else "", key='4'):
        nouveau_depositary = {"depositary": depositary_name}
        nouveau_compte = {"nom": new_account_name,
                          "solde": solde_initial_nouveau,
                          "taux_interet": taux_interet_nouveau,
                          "plafond": plafond_nouveau}
        save_saving_account(depositary_file, [nouveau_depositary, nouveau_compte])
        st.toast("Depositaire ajouté")

col1, col2, col3 = st.columns(3)
if len(json_files) > 0:
    with col1:
        st.subheader("📊 Savings information")
        barplot_savings(comptes)

        # Savings selection
        st.subheader("📌 Savings information")
        comptes = comptes[1:]
        comptes_sans_depositary = [{k: v for k, v in compte.items() if k != "depositary"} for compte in comptes]
        comptes_sans_depositary = [{**compte, "is_widget": True} for compte in comptes_sans_depositary]
        df = pd.DataFrame(comptes_sans_depositary)
        if len(comptes_sans_depositary) > 1:
            df = df.sort_values(by="nom")
        edited_df = st.data_editor(df,
                                   column_config={"nom": "Name saving",
                                                  "solde": "Sold (€)",
                                                  "taux_interet": "Interest rate (%)",
                                                  "plafond": "Saving limit (€)",
                                                  "is_widget": "Selection"},
                                   disabled=["nom", "solde", "taux_interet", "plafond"], hide_index=True)

        forcast = st.slider("**🗓️ Forcast (year)**", min_value=1, max_value=100, value=20, step=1)

    with col2:
        # Forcast savings and interest rate
        comptes_selectionnes = [compte for compte in comptes if
                                isinstance(compte, dict) and "nom" in compte and compte["nom"] in edited_df.loc[
                                    edited_df["is_widget"], "nom"].tolist()]

        montant_total = sum(compte["solde"] for compte in comptes_selectionnes)
        annee_en_cours = datetime.now().year
        calculer_soldes_previsionnels(comptes_selectionnes, forcast, annee_en_cours)
        interets_par_compte = calculer_interets(comptes_selectionnes, forcast, annee_en_cours)
        solde_total = calculer_solde_total(comptes_selectionnes, forcast)
        interets_total = calculer_interets_total(interets_par_compte, forcast)

        df = pd.DataFrame()
        title = "💰 Forcast savings"
        graphique_total(df, solde_total, forcast, annee_en_cours, comptes_selectionnes, title)

        df_interets = pd.DataFrame()
        title = "🪙 Forecast interest rate"
        graphique_interet(df_interets, interets_par_compte, forcast, interets_total, annee_en_cours, comptes_selectionnes, title)

    with col3:
        container = st.container()
        # Afficher la répartition optimale dans Streamlit
        st.header("Répartition Optimale des Fonds pour Maximiser les Intérêts")

        montant_total_initial = montant_total

        resultats = []
        nouveau_montant = 0

        # Boucle sur les années
        for annee in range(annee_en_cours, annee_en_cours + forcast + 1):
            # Optimiser les placements
            contraintes = [{"type": "eq", "fun": contrainte_somme,
                            "args": (montant_total_initial if annee == annee_en_cours else nouveau_montant,)}]
            bounds = [(0, compte["plafond"]) for compte in comptes_selectionnes]
            x0 = [0.0] * len(comptes_selectionnes)
            resultats_optimisation = minimize(
                objectif_placement,
                x0,
                args=(comptes_selectionnes,),
                constraints=contraintes,
                bounds=bounds
            )
            repartition_optimale = resultats_optimisation.x

            # Calculer les intérêts et mettre à jour les soldes
            nouveaux_soldes = []
            sold_a_placer = []
            nouveaux_interet = []
            if nouveau_montant < np.sum(compte["plafond"] for compte in comptes_selectionnes):
                for i, compte in enumerate(comptes_selectionnes):
                    interet = compte["taux_interet"] * repartition_optimale[i] / 100
                    repartition_optimale[i] += interet
                    nouveaux_soldes.append(repartition_optimale[i])
                    nouveaux_interet.append(interet)

            else:
                for i, compte in enumerate(comptes_selectionnes):
                    if ligne_resultat[f'{compte["nom"]} savings'] >= compte["plafond"]:
                        interet = compte["taux_interet"] * compte["plafond"] / 100
                    else:
                        interet = compte["taux_interet"] * ligne_resultat[f'{compte["nom"]} savings'] / 100
                    ligne_resultat[f'{compte["nom"]} savings'] += interet
                    nouveaux_soldes.append(ligne_resultat[f'{compte["nom"]} savings'])
                    nouveaux_interet.append(interet)

            # Nouveau montant pour l'année suivante
            nouveau_montant = np.sum(nouveaux_soldes)
            nouveau_montant_interet = np.sum(nouveaux_interet)

            # Enregistrer les résultats dans la liste
            ligne_resultat = {"Année": annee, "Total savings": nouveau_montant, "Total interest": nouveau_montant_interet}
            for i, compte in enumerate(comptes_selectionnes):
                ligne_resultat[f"{compte['nom']} savings"] = nouveaux_soldes[i]
                if (nouveaux_soldes[i] - nouveaux_interet[i])/compte["plafond"] < 0.95:
                    ligne_resultat[f"{compte['nom']} to_place"] = nouveaux_soldes[i] - nouveaux_interet[i]
                else:
                    ligne_resultat[f"{compte['nom']} to_place"] = f'Full ({compte["plafond"]})'
                ligne_resultat[f"{compte['nom']} interest"] = nouveaux_interet[i]
            resultats.append(ligne_resultat)

        # Créer le DataFrame résultat
        df_resultats = pd.DataFrame(resultats)

        # Créer un DataFrame pour les soldes
        df_solde = df_resultats.melt(id_vars=["Année"], var_name="Compte", value_name="savings")

        # Filtrer les colonnes contenant "Solde"
        df_solde = df_solde[df_solde["Compte"].str.contains("savings")]

        # Ajouter une colonne "Label" pour simplifier les étiquettes
        df_solde["Label"] = df_solde["Compte"].str.replace(" savings", "")

        opacity = alt.selection_single(fields=['Label'], on='click', bind='legend')
        chart_solde = alt.Chart(df_solde).mark_line().encode(
            x="Année",
            y=alt.Y("savings:Q", axis=alt.Axis(title="Montant Total + Solde")),
            color=alt.Color("Label:N", legend=alt.Legend(title=None)),
            tooltip=["Année", "savings"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
        ).interactive() + alt.Chart(df_solde).mark_circle(size=100).encode(
            x="Année",
            y=alt.Y("savings:Q", axis=alt.Axis(title="Montant Total + Solde")),
            color=alt.Color("Label:N", legend=alt.Legend(title=None)),
            tooltip=["Année", "savings"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
        ).properties(
            width=600,
            height=400
        ).add_params(opacity)

        # Afficher le graphique avec Streamlit
        container.altair_chart(chart_solde, theme=None, use_container_width=True)

        # Créer un DataFrame pour les intérêts
        df_interet = df_resultats.melt(id_vars=["Année"], var_name="Compte", value_name="interest")

        # Filtrer les colonnes contenant "Intérêt"
        df_interet = df_interet[df_interet["Compte"].str.contains("interest")]

        # Ajouter une colonne "Label" pour simplifier les étiquettes
        df_interet["Label"] = df_interet["Compte"].str.replace(" interest", "")

        opacity = alt.selection_single(fields=['Label'], on='click', bind='legend')
        chart_interet = alt.Chart(df_interet).mark_line().encode(
            x="Année",
            y=alt.Y("interest:Q"),
            color=alt.Color("Label:N", legend=alt.Legend(title=None)),
            tooltip=["Année", "interest"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
        ).interactive() + alt.Chart(df_interet).mark_circle(size=100).encode(
            x="Année",
            y=alt.Y("interest:Q",),
            color=alt.Color("Label:N", legend=alt.Legend(title=None)),
            tooltip=["Année", "interest"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
        ).properties(
            width=600,
            height=400
        ).add_params(opacity)

        # Afficher le graphique avec Streamlit
        container.altair_chart(chart_interet, theme=None, use_container_width=True)

    st.dataframe(df_resultats, hide_index=True, use_container_width=True)
else:
    st.subheader("Welcome to GK!LB. A little FinTech app")
    st.subheader("For start, add your first depositary with the left panel and don't forget to update")
