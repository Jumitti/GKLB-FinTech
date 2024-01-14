import json
import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize


# Depositary folder
def get_depositary_name(depositary_file):
    with open(os.path.join(depositary_path, depositary_file), 'r') as json_file:
        data = json.load(json_file)
        return data[0].get("depositary", "")


# Loading depositary information
def load_saving_accounts(depositary_file):
    try:
        with open(f"depositary/{depositary_file}", "r") as depositary_file:
            saving_accounts = json.load(depositary_file)
    except FileNotFoundError:
        saving_accounts = []
    return saving_accounts


# Update/create depositary information
def save_saving_account(depositary_file, savings):
    with open(f"depositary/{depositary_file}", "w") as depositary_file:
        json.dump(savings, depositary_file, indent=2)


# Barplot savings
def barplot_savings(savings):
    df = pd.DataFrame(savings)

    df = df.rename(columns={'saving': 'Name saving'})
    df = df.rename(columns={'sold': 'Sold (€)'})
    df = df.rename(columns={'interest_rate': 'Interest rate (%)'})
    df = df.rename(columns={'limit': 'Limit (€)'})
    df = pd.concat([df, pd.DataFrame([{"Name saving": "Total", "Sold (€)": df['Sold (€)'].sum()}])])  # Total of savings

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Name saving:N', axis=alt.Axis(title=None)),
        y=alt.Y('Sold (€):Q', axis=alt.Axis(title='Sold (€)')),
        color='Name saving',
        tooltip=['Name saving:N', 'Sold (€):Q', 'Limit (€):Q', 'Interest rate (%):Q'],
    ) + alt.Chart(df).mark_bar().encode(
        x=alt.X('Name saving:N', axis=alt.Axis(title=None)),
        y=alt.Y('Limit (€):Q', axis=alt.Axis(title='Limit (€)')),
        color='Name saving',
        opacity=alt.value(0.5),
        tooltip=['Name saving:N', 'Sold (€):Q', 'Limit (€):Q', 'Interest rate (%):Q']
    ).properties(
        width=400,
        height=300
    ).interactive()

    st.altair_chart(chart, theme=None, use_container_width=True)


# Savings evolution
def forecast_sold(savings, forecast, current_year):
    for saving in savings:
        forecast_sold = saving["sold"]
        forecast_sold_list = [forecast_sold]

        for year in range(1, forecast + 1):
            annual_sold = min(forecast_sold, saving["limit"])
            annual_interest = annual_sold * (saving["interest_rate"] / 100)
            forecast_sold += annual_interest
            forecast_sold_list.append(forecast_sold)

        saving["forecast_sold"] = forecast_sold_list
        saving["current_year"] = current_year


# Total of savings evolution
def total_sold(savings, forecast):
    total_sold = [0] * (forecast + 1)

    for saving in savings:
        total_sold = [sold + saving_sold for sold, saving_sold in zip(total_sold, saving["forecast_sold"])]

    return total_sold


# Graphic of savings
def sold_chart(df_sold, total_sold, forecast, current_year, selected_savings, title):
    df_sold["Year"] = list(range(current_year, current_year + forecast + 1))
    df_sold["Savings"] = "Total"
    df_sold["Sold (€)"] = total_sold

    for saving in selected_savings:
        df_params = pd.DataFrame({
            'Year': list(range(current_year, current_year + forecast + 1)),
            'Savings': saving["saving"],
            'Sold (€)': saving["forecast_sold"]
        })
        df_sold = pd.concat([df_sold, df_params], ignore_index=True)

    st.subheader(title)
    opacity = alt.selection_single(fields=['Savings'], on='click', bind='legend')
    sold_chart = alt.Chart(df_sold).mark_line().encode(
        x='Year',
        y='Sold (€)',
        color='Savings', opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ) + alt.Chart(df_sold).mark_circle(size=100).encode(
        x='Year',
        y='Sold (€)',
        color='Savings',
        tooltip=['Savings', 'Sold (€)', 'Year'], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ).properties(
        width=600,
        height=400
    ).interactive().add_params(opacity)

    st.altair_chart(sold_chart, theme=None, use_container_width=True)


# Interest rate evolution
def forecast_interest(savings, forecast):
    saving_interest = {saving["saving"]: [0] * (forecast + 1) for saving in savings}

    for year in range(1, forecast + 1):
        for saving in savings:
            forecast_sold = min(saving["forecast_sold"][year - 1], saving["limit"])
            annual_interest = forecast_sold * (saving["interest_rate"] / 100)
            saving_interest[saving["saving"]][year] = annual_interest

    return saving_interest


# Total of interest rate evolution
def total_interest(saving_interest, forecast):
    total_interest = [0] * (forecast + 1)

    for saving, interest in saving_interest.items():
        for year in range(forecast + 1):
            total_interest[year] += interest[year]

    return total_interest


# Graphic of interest rate
def interest_chart(df_interest, saving_interest, forecast, total_interest, current_year, selected_savings, title):
    df_interest["Year"] = list(range(current_year, current_year + forecast + 1))
    df_interest["Savings"] = "Total"
    df_interest["Interest (€)"] = total_interest

    for saving in selected_savings:
        df_params = pd.DataFrame({
            'Year': list(range(current_year, current_year + forecast + 1)),
            'Savings': saving["saving"],
            'Interest (€)': saving_interest[saving["saving"]]
        })
        df_interest = pd.concat([df_interest, df_params], ignore_index=True)

    st.subheader(title)
    opacity = alt.selection_single(fields=['Savings'], on='click', bind='legend')
    interest_chart = alt.Chart(df_interest).mark_line().encode(
        x='Year',
        y='Interest (€)',
        color='Savings', opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ) + alt.Chart(df_interest).mark_circle(size=100).encode(
        x='Year',
        y='Interest (€)',
        color='Savings',
        tooltip=['Savings', 'Interest (€)', 'Year'], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ).properties(
        width=600,
        height=400
    ).interactive().add_params(opacity)

    st.altair_chart(interest_chart, theme=None, use_container_width=True)


# Maximizing investments
def placement_optimization(x, savings):
    return -sum(saving["interest_rate"] * xi for saving, xi in zip(savings, x))


# Contrainte de somme totale
def limit_constraint(x, total_sold):
    return sum(x) - total_sold


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
depositary_path = "depositary"
depositary_files = [depositary_file for depositary_file in os.listdir(depositary_path) if
                    depositary_file.endswith(".json")]

if len(depositary_files) > 0:
    depositary_mapping = {}
    for depositary_file in depositary_files:
        depositary_file_path = os.path.join(depositary_path, depositary_file)

        try:
            with open(depositary_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
                if isinstance(data, list) and data and isinstance(data[0], dict) and "depositary" in data[0]:
                    depositary = data[0]["depositary"]
                    depositary_mapping[depositary] = depositary_file
                else:
                    st.error(f"File {depositary_file} does not have the expected structure.")
        except Exception as e:
            st.error(f"Error reading file {depositary_file}: {e}")

    # Depositary selection
    selected_depositary_file = st.sidebar.selectbox("Sélectionnez le dépositaire :", list(depositary_mapping.keys()))
    depositary_file = depositary_mapping[selected_depositary_file]
    savings = load_saving_accounts(depositary_file)

    # Add a saving account
    with st.sidebar.expander("Add a saving account", expanded=False):
        saving_name = st.text_input("Name")
        initial_sold = st.number_input("Solde initial du nouveau compte", step=1.0, min_value=0.0)
        interest_rate = st.number_input("Taux d'intérêt (%) du nouveau compte", step=0.1, min_value=0.0,
                                        max_value=100.0)
        limit = st.number_input("Plafond du nouveau compte", step=1.0, min_value=0.0)

        if any(isinstance(saving, dict) and "saving" in saving and saving["saving"] == saving_name for saving in
               savings):
            disable = True
        else:
            disable = False

        if st.button("Ajouter le nouveau compte", disabled=disable,
                     help="Le nom existe déjà" if disable else ""):
            new_saving = {
                "saving": saving_name,
                "sold": initial_sold,
                "interest_rate": interest_rate,
                "limit": limit
            }
            savings.append(new_saving)
            save_saving_account(depositary_file, savings)
            st.toast("Nouveau compte")

    # Update a saving account
    with st.sidebar.expander("Modifier un compte", expanded=False):
        selected_saving = st.selectbox(
            "Sélectionner un compte",
            [saving["saving"] for saving in savings if isinstance(saving, dict) and "saving" in saving],
            index=0)

        if selected_saving:
            selected_saving = next((saving for saving in savings if
                                    isinstance(saving, dict) and "saving" in saving and saving[
                                        "saving"] == selected_saving), None)
            if selected_saving:
                initial_sold = st.number_input("Solde initial", value=float(selected_saving["sold"]), step=1.0,
                                               min_value=0.0)
                interest_rate = st.number_input("Taux d'intérêt (%)", value=float(selected_saving["interest_rate"]),
                                                step=0.1, min_value=0.0, max_value=100.0)
                limit = st.number_input("Plafond du compte", value=float(selected_saving["limit"]), step=1.0,
                                        min_value=0.0)
                selected_saving["sold"] = initial_sold
                selected_saving["interest_rate"] = interest_rate
                selected_saving["limit"] = limit
                save_saving_account(depositary_file, savings)

                # Delete saving account
                if st.button("Supprimer le compte"):
                    savings.remove(selected_saving)
                    save_saving_account(depositary_file, savings)
                    st.toast("Compte modifié")

    # Delete a depositary
    with st.sidebar.expander("Supprimer le dépositaire"):
        st.warning(f"Êtes-vous sûr de vouloir supprimer le dépositaire '{selected_depositary_file}'?")
        if st.button("Oui"):
            os.remove(os.path.join(depositary_path, depositary_file))
            st.toast(f"Le dépositaire '{selected_depositary_file}' a été supprimé avec succès.")

    st.sidebar.divider()

# Create a depositary
with st.sidebar.expander("Ajouter un dépositaire", expanded=False):
    depositary_name = st.text_input("Nom du dépositaire")
    saving_name = st.text_input("Nom du nouveau compte")
    initial_sold = st.number_input("Solde initial du nouveau compte", step=1.0, min_value=0.0, key='1')
    interest_rate = st.number_input("Taux d'intérêt (%) du nouveau compte", step=0.1, min_value=0.0, max_value=100.0,
                                    key='2')
    limit = st.number_input("Plafond du nouveau compte", step=1.0, min_value=0.0, key='3')

    depositary_file = f"depositary_{depositary_name}.json"
    if len(depositary_files) > 0:
        if os.path.exists(f'depositary/{depositary_file}'):
            disable = True
        else:
            disable = False
    else:
        disable = False

    if st.button("Ajouter le nouveau compte", disabled=disable,
                 help="Le nom existe déjà" if disable else "", key='4'):
        new_depositary = {"depositary": depositary_name}
        new_saving = {"saving": saving_name,
                      "sold": initial_sold,
                      "interest_rate": interest_rate,
                      "limit": limit}
        save_saving_account(depositary_file, [new_depositary, new_saving])
        st.toast("Depositaire ajouté")

col1, col2, col3 = st.columns([0.9, 1.1, 1.1])
if len(depositary_files) > 0:
    with col1:
        st.subheader("📊 Savings information")
        barplot_savings(savings)

        # Savings selection
        st.subheader("📌 Savings information")
        savings = savings[1:]
        exclude_depositary = [{k: v for k, v in saving.items() if k != "depositary"} for saving in savings]
        exclude_depositary = [{"is_widget": True, **depositary} for depositary in exclude_depositary]
        df_depositary_savings = pd.DataFrame(exclude_depositary)
        if len(exclude_depositary) > 1:
            df_depositary_savings = df_depositary_savings.sort_values(by="saving")
        depositary_savings_selection = st.data_editor(df_depositary_savings,
                                                      column_config={"is_widget": "Selection",
                                                                     "saving": "Name saving",
                                                                     "sold": "Sold (€)",
                                                                     "interest_rate": "Interest rate (%)",
                                                                     "limit": "Saving limit (€)",
                                                                     },
                                                      disabled=["saving", "sold", "interest_rate", "limit"],
                                                      hide_index=True)

        forecast = st.slider("**🗓️ Forecast (year)**", min_value=1, max_value=100, value=20, step=1)

    with col2:
        # forecast savings and interest rate
        selected_savings = [saving for saving in savings if
                            isinstance(saving, dict) and "saving" in saving and saving["saving"] in
                            depositary_savings_selection.loc[
                                depositary_savings_selection["is_widget"], "saving"].tolist()]

        current_year = datetime.now().year
        forecast_sold(selected_savings, forecast, current_year)
        saving_interest = forecast_interest(selected_savings, forecast)
        total_sold = total_sold(selected_savings, forecast)
        total_interest = total_interest(saving_interest, forecast)

        df_sold = pd.DataFrame()
        title = "💰 Forecast savings"
        sold_chart(df_sold, total_sold, forecast, current_year, selected_savings, title)

        df_interest = pd.DataFrame()
        title = "🪙 Forecast interest rate"
        interest_chart(df_interest, saving_interest, forecast, total_interest, current_year,
                       selected_savings, title)

    with col3:
        container = st.container()
        # Afficher la répartition optimale dans Streamlit
        total_sold = sum(saving["sold"] for saving in selected_savings)
        optimized_forecast = []
        optimized_total_sold = 0

        # Boucle sur les années
        for year in range(current_year, current_year + forecast + 1):
            # Optimiser les placements
            constraint = [{"type": "eq", "fun": limit_constraint,
                           "args": (total_sold if year == current_year else optimized_total_sold,)}]
            bounds = [(0, saving["limit"]) for saving in selected_savings]
            x0 = [0.0] * len(selected_savings)
            placements_optimization = minimize(
                placement_optimization,
                x0,
                args=(selected_savings,),
                constraints=constraint,
                bounds=bounds
            )
            optimized_placements = placements_optimization.x

            # Calculer les intérêts et mettre à jour les soldes
            optimized_sold = []
            optimized_interest = []
            if optimized_total_sold < np.sum(saving["limit"] for saving in selected_savings):
                for i, saving in enumerate(selected_savings):
                    interet = saving["interest_rate"] * optimized_placements[i] / 100
                    optimized_placements[i] += interet
                    optimized_sold.append(optimized_placements[i])
                    optimized_interest.append(interet)

            else:
                for i, saving in enumerate(selected_savings):
                    if data_optimized[f'{saving["saving"]} sold'] >= saving["limit"]:
                        interet = saving["interest_rate"] * saving["limit"] / 100
                    else:
                        interet = saving["interest_rate"] * data_optimized[f'{saving["saving"]} sold'] / 100
                    data_optimized[f'{saving["saving"]} sold'] += interet
                    optimized_sold.append(data_optimized[f'{saving["saving"]} sold'])
                    optimized_interest.append(interet)

            # Nouveau montant pour l'année suivante
            optimized_total_sold = np.sum(optimized_sold)
            optimized_total_interest = np.sum(optimized_interest)

            # Enregistrer les résultats dans la liste
            data_optimized = {"Year": year, "Total sold": optimized_total_sold,
                              "Total interest": optimized_total_interest}
            for i, saving in enumerate(selected_savings):
                data_optimized[f"{saving['saving']} sold"] = optimized_sold[i]
                if (optimized_sold[i] - optimized_interest[i]) / saving["limit"] < 0.95:
                    data_optimized[f"{saving['saving']} to_place"] = optimized_sold[i] - optimized_interest[i]
                else:
                    data_optimized[f"{saving['saving']} to_place"] = f'**Full** ({saving["limit"]})'
                data_optimized[f"{saving['saving']} interest"] = optimized_interest[i]
            optimized_forecast.append(data_optimized)

        df_optimized_forecast = pd.DataFrame(optimized_forecast)

        df_optimized_forecast_sold = df_optimized_forecast.melt(id_vars=["Year"], var_name="saving", value_name="sold")
        df_optimized_forecast_sold = df_optimized_forecast_sold[
            df_optimized_forecast_sold["saving"].str.contains("sold")]
        df_optimized_forecast_sold["Savings"] = df_optimized_forecast_sold["saving"].str.replace(" sold", "")
        df_optimized_forecast_sold = df_optimized_forecast_sold.rename(columns={'sold': 'Sold (€)'})

        opacity = alt.selection_single(fields=['Savings'], on='click', bind='legend')
        optimized_sold_chart = alt.Chart(df_optimized_forecast_sold).mark_line().encode(
            x="Year",
            y="Sold (€):Q",
            color="Savings:N", opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
        ) + alt.Chart(df_optimized_forecast_sold).mark_circle(size=100).encode(
            x=alt.X("Year", axis=alt.Axis(title="Year")),
            y="Sold (€):Q",
            color="Savings:N",
            tooltip=['Savings:N', "Sold (€)", "Year"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
        ).properties(
            width=600,
            height=400
        ).interactive().add_params(opacity)

        container.subheader('💫💰 Optimized forecast savings')
        container.altair_chart(optimized_sold_chart, theme=None, use_container_width=True)

        df_optimized_forecast_interest = df_optimized_forecast.melt(id_vars=["Year"], var_name="saving",
                                                                    value_name="interest")
        df_optimized_forecast_interest = df_optimized_forecast_interest[
            df_optimized_forecast_interest["saving"].str.contains("interest")]
        df_optimized_forecast_interest["Savings"] = df_optimized_forecast_interest["saving"].str.replace(" interest",
                                                                                                         "")
        df_optimized_forecast_interest = df_optimized_forecast_interest.rename(columns={'interest': 'Interest (€)'})

        opacity = alt.selection_single(fields=['Savings'], on='click', bind='legend')
        optimized_interest_chart = alt.Chart(df_optimized_forecast_interest).mark_line().encode(
            x="Year",
            y="Interest (€):Q",
            color="Savings:N",
            tooltip=["Year", "Interest (€)"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
        ).interactive() + alt.Chart(df_optimized_forecast_interest).mark_circle(size=100).encode(
            x="Year",
            y="Interest (€):Q",
            color="Savings:N",
            tooltip=["Year", "Interest (€)"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
        ).properties(
            width=600,
            height=400
        ).add_params(opacity)

        container.subheader('💫🪙 Optimized forecast interest rate')
        container.altair_chart(optimized_interest_chart, theme=None, use_container_width=True)

    st.subheader("💫🧮 Optimal forecast placements")
    subset_sold = pd.IndexSlice[:, df_optimized_forecast.columns[df_optimized_forecast.columns.str.contains('sold')]]
    subset_to_place = pd.IndexSlice[:, df_optimized_forecast.columns[df_optimized_forecast.columns.str.contains('to_place')]]
    subset_interest = pd.IndexSlice[:, df_optimized_forecast.columns[df_optimized_forecast.columns.str.contains('interest')]]
    styled_df = df_optimized_forecast.style.set_properties(**{'background-color': '#FFCCCC'}, subset=subset_sold) \
        .set_properties(**{'background-color': '#CCCCFF'}, subset=subset_to_place) \
        .set_properties(**{'background-color': '#CCFFCC'}, subset=subset_interest)
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

else:
    st.subheader("Welcome to GK!LB. A little FinTech app")
    st.subheader("For start, add your first depositary with the left panel and don't forget to update")
