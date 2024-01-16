# MIT License
#
# Copyright (c) 2024 Minniti Julien
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize


# Loading depositary information
def load_saving_accounts(depositary_path, depositary_file):
    try:
        with open(f"{depositary_path}/{depositary_file}", "r") as depositary_file:
            saving_accounts = json.load(depositary_file)
    except FileNotFoundError:
        saving_accounts = []
    return saving_accounts


# Update/create depositary information
def save_saving_account(depositary_path, depositary_file, savings):
    with open(f"{depositary_path}/{depositary_file}", "w") as depositary_file:
        json.dump(savings, depositary_file, indent=2)


def contains_special_characters(text):
    forbidden_characters = r"\/:*?\"<>|"
    return any(char in forbidden_characters for char in text)


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
        width=600,
        height=400
    ).interactive()

    st.altair_chart(chart, theme=None, use_container_width=True)


# Savings evolution
def forecast_sold(savings, forecast, current_year):
    for saving in savings:
        forecast_sold = saving["sold"]
        forecast_sold_list = [forecast_sold]

        for year in range(1, forecast + 1):
            if saving["limit"] == 0:
                annual_sold = forecast_sold
            else:
                annual_sold = min(forecast_sold, saving["limit"])
            annual_interest = annual_sold * (saving["interest_rate"] / 100)
            forecast_sold += annual_interest
            forecast_sold_list.append(forecast_sold)

        saving["forecast_sold"] = forecast_sold_list
        saving["current_year"] = 2023


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
    opacity = alt.selection_point(fields=['Savings'], on='click', bind='legend')
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

    return df_sold


# Interest rate evolution
def forecast_interest(savings, forecast):
    saving_interest = {saving["saving"]: [0] * (forecast + 1) for saving in savings}

    for year in range(forecast + 1):
        for saving in savings:
            if saving["limit"] == 0:
                forecast_sold = saving["forecast_sold"][year - 1]
            else:
                forecast_sold = min(saving["forecast_sold"][year - 1], saving["limit"])
            annual_interest = forecast_sold * (saving["interest_rate"] / 100)
            saving_interest[saving["saving"]][year - 1] = annual_interest

    return saving_interest


# Total of interest rate evolution
def total_interest(saving_interest, forecast):
    total_interest = [0] * (forecast + 1)

    for saving, interest in saving_interest.items():
        for year in range(1, forecast + 2):
            total_interest[year - 1] += interest[year - 1]

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
    opacity = alt.selection_point(fields=['Savings'], on='click', bind='legend')
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

    return df_interest


def optimized_sold_chart(df_optimized_forecast, title, yaxis):
    df_optimized_forecast_sold = df_optimized_forecast.melt(id_vars=["Year"], var_name="saving", value_name="sold")
    df_optimized_forecast_sold = df_optimized_forecast_sold[
        df_optimized_forecast_sold["saving"].str.contains("sold")]
    df_optimized_forecast_sold["Savings"] = df_optimized_forecast_sold["saving"].str.replace(" sold", "")
    df_optimized_forecast_sold = df_optimized_forecast_sold.rename(columns={'sold': 'Sold (€)'})

    opacity = alt.selection_point(fields=['Savings'], on='click', bind='legend')
    optimized_sold_chart = alt.Chart(df_optimized_forecast_sold).mark_line().encode(
        x="Year",
        y=alt.Y("Sold (€):Q", axis=alt.Axis(title=yaxis)),
        color="Savings:N", opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ) + alt.Chart(df_optimized_forecast_sold).mark_circle(size=100).encode(
        x=alt.X("Year", axis=alt.Axis(title="Year")),
        y=alt.Y("Sold (€):Q", axis=alt.Axis(title=yaxis)),
        color="Savings:N",
        tooltip=['Savings:N', "Sold (€)", "Year"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ).properties(
        width=600,
        height=400
    ).interactive().add_params(opacity)

    st.subheader(title)
    st.altair_chart(optimized_sold_chart, theme=None, use_container_width=True)


def optimized_interest_chart(df_optimized_forecast, title, yaxis):
    df_optimized_forecast_interest = df_optimized_forecast.melt(id_vars=["Year"], var_name="saving",
                                                                value_name="interest")
    df_optimized_forecast_interest = df_optimized_forecast_interest[
        df_optimized_forecast_interest["saving"].str.contains("interest")]
    df_optimized_forecast_interest["Savings"] = df_optimized_forecast_interest["saving"].str.replace(" interest",
                                                                                                     "")
    df_optimized_forecast_interest = df_optimized_forecast_interest.rename(columns={'interest': 'Interest (€)'})

    opacity = alt.selection_point(fields=['Savings'], on='click', bind='legend')
    optimized_interest_chart = alt.Chart(df_optimized_forecast_interest).mark_line().encode(
        x="Year",
        y=alt.Y("Interest (€):Q", axis=alt.Axis(title=yaxis)),
        color="Savings:N",
        tooltip=["Year", "Interest (€)"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ).interactive() + alt.Chart(df_optimized_forecast_interest).mark_circle(size=100).encode(
        x="Year",
        y=alt.Y("Interest (€):Q", axis=alt.Axis(title=yaxis)),
        color="Savings:N",
        tooltip=["Year", "Interest (€)"], opacity=alt.condition(opacity, alt.value(0.8), alt.value(0.2))
    ).properties(
        width=600,
        height=400
    ).add_params(opacity)

    st.subheader(title)
    st.altair_chart(optimized_interest_chart, theme=None, use_container_width=True)


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

# Sidebar manager
st.sidebar.image("https://raw.githubusercontent.com/Jumitti/GKLB-FinTech/master/img/GKLB_wobg.png", use_column_width=True)
st.sidebar.title(f"[GK!LB - Manager](https://www.youtube.com/watch?v=S4Ez-aDbAoA)")

if st.sidebar.button("🔄️ Update"):  # Update info
    st.toast("Updated", icon="🔄️")
st.sidebar.divider()

# Load all depositary
current_directory = os.path.dirname(os.path.abspath(__file__))
depositary_path = os.path.join(current_directory, "depositary")
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
    selected_depositary_file = st.sidebar.selectbox("Select depositary:", list(depositary_mapping.keys()))
    depositary_file = depositary_mapping[selected_depositary_file]
    savings = load_saving_accounts(depositary_path, depositary_file)

    # Add a saving account
    with st.sidebar.expander("Add saving account", expanded=False):
        saving_name = st.text_input("Saving account name", max_chars=20, help=' \ / : * ? \ " < > | not allowed')
        initial_sold = st.number_input("Initial sold (€)", step=1.0, min_value=0.0)
        interest_rate = st.number_input("Interest rate (%)", step=0.1, min_value=0.0,
                                        max_value=100.0)
        limit = st.number_input("Limit (€)", step=1.0, min_value=0.0, help="For no limit, use 0")

        if any(isinstance(saving, dict) and "saving" in saving and saving["saving"] == saving_name for saving in
               savings):
            disable = True
            disable_message = "Saving account already exists"
        else:
            disable = False

        if contains_special_characters(saving_name):
            disable = True
            disable_message = ' \ / : * ? \ " < > | not allowed'
        else:
            disable = False

        if st.button("Add saving account", disabled=disable,
                     help=disable_message if disable else ""):
            new_saving = {
                "saving": saving_name,
                "sold": initial_sold,
                "interest_rate": interest_rate,
                "limit": limit
            }
            savings.append(new_saving)
            save_saving_account(depositary_path, depositary_file, savings)
            st.toast(f"Saving account **{saving_name}** created", icon='💰')

    # Update a saving account
    with st.sidebar.expander("Edit saving account", expanded=False):
        selected_saving = st.selectbox(
            "Select saving account",
            [saving["saving"] for saving in savings if isinstance(saving, dict) and "saving" in saving],
            index=0)

        if selected_saving:
            selected_saving = next((saving for saving in savings if
                                    isinstance(saving, dict) and "saving" in saving and saving[
                                        "saving"] == selected_saving), None)
            if selected_saving:
                initial_sold = st.number_input("Sold (€)", value=float(selected_saving["sold"]), step=1.0,
                                               min_value=0.0)
                interest_rate = st.number_input("Interest rate (%)", value=float(selected_saving["interest_rate"]),
                                                step=0.1, min_value=0.0, max_value=100.0)
                limit = st.number_input("Limit (€)", value=float(selected_saving["limit"]), step=1.0,
                                        min_value=0.0, help="For no limit, use 0")
                selected_saving["sold"] = initial_sold
                selected_saving["interest_rate"] = interest_rate
                selected_saving["limit"] = limit
                save_saving_account(depositary_path, depositary_file, savings)

                # Delete saving account
                if len(savings) == 2:
                    disable_delete = True
                else:
                    disable_delete = False
                if st.button("Delete saving account", disabled=disable_delete,
                             help="Cannot delete an account if the depository only has one" if disable_delete else ""):
                    savings.remove(selected_saving)
                    save_saving_account(depositary_path, depositary_file, savings)
                    st.toast(f"Saving account **{selected_saving['saving']}** deleted", icon='🗑️')

    # Delete a depositary
    with st.sidebar.expander("Delete depositary"):
        st.warning(f"Are you sure you want to remove depositary **{selected_depositary_file}**?")
        if st.button("Yes"):
            os.remove(os.path.join(depositary_path, depositary_file))
            st.toast(f"Depositary **{selected_depositary_file}** deleted", icon='🗑️')
    st.sidebar.divider()

# Create a depositary
with st.sidebar.expander("Create depositary profile", expanded=False):
    depositary_name = st.text_input("Depositary name", max_chars=20, help=' \ / : * ? \ " < > | not allowed')
    saving_name = st.text_input("Saving account name", max_chars=20, key="4", help=' \ / : * ? \ " < > | not allowed')
    initial_sold = st.number_input("Initial sold (€)", step=1.0, min_value=0.0, key='1')
    interest_rate = st.number_input("Interest rate (%)", step=0.1, min_value=0.0, max_value=100.0, key='2')
    limit = st.number_input("Limit (€)", step=1.0, min_value=0.0, key='3', help="For no limit, use 0")

    depositary_file = f"depositary_{depositary_name}.json"
    if len(depositary_files) > 0:
        if os.path.exists(f'depositary/{depositary_file}'):
            disable = True
            disable_message = "Depositary name already exists"
        else:
            disable = False
    else:
        disable = False

    if contains_special_characters(depositary_name) or contains_special_characters(saving_name):
        disable = True
        disable_message = ' \ / : * ? \ " < > | not allowed'
    else:
        disable = False

    if st.button("Create depositary profile", disabled=disable,
                 help=disable_message if disable else ""):
        new_depositary = {"depositary": depositary_name}
        new_saving = {"saving": saving_name,
                      "sold": initial_sold,
                      "interest_rate": interest_rate,
                      "limit": limit}
        save_saving_account(depositary_path, depositary_file, [new_depositary, new_saving])
        st.toast(f"Depositary **{depositary_name}** created", icon='🎉')

col1, col2, col3 = st.columns(3)
with col1:
    # Main page
    st.title('📈 💸 GK!LB - FinTech')
    st.markdown(
        "<div style='text-align: justify;'><strong>GK!LB</strong> is a comprehensive financial tool designed to monitor the performance of savings accounts across "
        "various depositories. With a user-friendly interface, users can seamlessly optimize their investments "
        "by efficiently managing multiple savings accounts within a single application. The left sidebar "
        "facilitates easy addition of depositories and provides convenient controls for configuring savings "
        "accounts—enabling users to add, modify, or delete accounts with ease. GK!LB simplifies the process of "
        "tracking and optimizing savings, offering a streamlined and efficient financial management solution.</div>",
        unsafe_allow_html=True)
    st.write(
        "Created by Minniti Julien - [GitHub](https://github.com/Jumitti/GKLB-FinTech) - [Download app](https://github.com/Jumitti/GKLB-FinTech/releases/tag/beta_v0.1) - [MIT licence](https://github.com/Jumitti/GKLB-FinTech/blob/master/LICENSE)")
    st.divider()
if len(depositary_files) > 0:
    with col2:
        st.subheader("📊 Savings information")
        barplot_savings(savings)

    with col3:
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
                                                      hide_index=True, use_container_width=True)

    with col1:
        forecast = st.slider("**🗓️ Forecast (year)**", min_value=1, max_value=100, value=20, step=1)

    st.divider()
    col1b, col2b, col3b = st.columns(3, gap="small")
    with col1b:
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
        title = "💸💰 Forecast savings"
        df_all_sold = sold_chart(df_sold, total_sold, forecast, current_year, selected_savings, title)

        df_interest = pd.DataFrame()
        title = "💸🪙 Forecast interests"
        df_all_interest = interest_chart(df_interest, saving_interest, forecast, total_interest, current_year,
                                         selected_savings, title)

    with col2b:
        # Afficher la répartition optimale dans Streamlit
        total_sold = sum(saving["sold"] for saving in selected_savings)
        optimized_forecast = []
        optimized_total_sold = 0

        if len(selected_savings) >= 2:
            # Boucle sur les années
            for year in range(current_year, current_year + forecast + 2):
                # Optimiser les placements
                constraint = [{"type": "eq", "fun": limit_constraint,
                               "args": (total_sold if year == current_year else optimized_total_sold,)}]
                bounds = [(0, float('inf')) if saving["limit"] == 0 else (0, saving["limit"]) for saving in
                          selected_savings]
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
                if optimized_total_sold < np.sum(list(saving["limit"] for saving in selected_savings)) or any(saving["limit"] == 0 for saving in selected_savings):
                    for i, saving in enumerate(selected_savings):
                        interest = saving["interest_rate"] * optimized_placements[i] / 100
                        optimized_placements[i] += interest
                        optimized_sold.append(optimized_placements[i])
                        optimized_interest.append(interest)

                else:
                    for i, saving in enumerate(selected_savings):
                        if data_optimized[f'{saving["saving"]} sold'] >= saving["limit"] != 0:
                            interest = saving["interest_rate"] * saving["limit"] / 100
                        else:
                            interest = saving["interest_rate"] * data_optimized[f'{saving["saving"]} sold'] / 100
                        data_optimized[f'{saving["saving"]} sold'] += interest
                        optimized_sold.append(data_optimized[f'{saving["saving"]} sold'])
                        optimized_interest.append(interest)

                # Nouveau montant pour l'année suivante
                optimized_total_sold = np.sum(optimized_sold)
                optimized_total_interest = np.sum(optimized_interest)

                # Enregistrer les résultats dans la liste
                data_optimized = {"Year": year, "Total sold": optimized_total_sold - optimized_total_interest,
                                  "Total interest": optimized_total_interest}
                for i, saving in enumerate(selected_savings):
                    data_optimized[f"{saving['saving']} sold"] = optimized_sold[i]
                    if saving["limit"] == 0:
                        data_optimized[f"{saving['saving']} to_place"] = optimized_sold[i] - optimized_interest[i]
                    else:
                        if (optimized_sold[i] - optimized_interest[i]) / saving["limit"] < 0.95:
                            data_optimized[f"{saving['saving']} to_place"] = optimized_sold[i] - optimized_interest[i]
                        else:
                            data_optimized[f"{saving['saving']} to_place"] = saving["limit"]
                    data_optimized[f"{saving['saving']} interest"] = optimized_interest[i]
                optimized_forecast.append(data_optimized)

            df_optimized_forecast = pd.DataFrame(optimized_forecast)

            title = '💫💰 Optimized forecast savings'
            optimized_sold_chart(df_optimized_forecast, title, yaxis='Sold (€)')

            title = '💫🪙 Optimized forecast interests'
            optimized_interest_chart(df_optimized_forecast, title, yaxis='Interest (€)')

    if len(selected_savings) >= 2:
        st.subheader("💫🧮 Optimal forecast placements")
        subset_sold = pd.IndexSlice[:, df_optimized_forecast.columns[df_optimized_forecast.columns.str.contains('sold')]]
        subset_to_place = pd.IndexSlice[:, df_optimized_forecast.columns[df_optimized_forecast.columns.str.contains('to_place')]]
        subset_interest = pd.IndexSlice[:, df_optimized_forecast.columns[df_optimized_forecast.columns.str.contains('interest')]]
        styled_df = df_optimized_forecast.style.set_properties(**{'background-color': '#FFCCCC'}, subset=subset_sold) \
            .set_properties(**{'background-color': '#CCCCFF'}, subset=subset_to_place) \
            .set_properties(**{'background-color': '#CCFFCC'}, subset=subset_interest)
        st.dataframe(styled_df, hide_index=True, use_container_width=True)

        with col3b:
            df1 = df_all_sold
            df1['Year'] = pd.to_numeric(df1['Year'].replace(',', ''), errors='coerce')
            df_transformed = df1.pivot(index='Year', columns='Savings', values='Sold (€)')
            df_transformed = df_transformed.fillna(0)
            df_transformed.columns = [f'{col} sold' if col != 'Year' else col for col in df_transformed.columns]
            df1 = df_transformed.reset_index()

            df2 = df_all_interest
            df2['Year'] = pd.to_numeric(df2['Year'].replace(',', ''), errors='coerce')
            df_transformed = df2.pivot(index='Year', columns='Savings', values='Interest (€)')
            df_transformed = df_transformed.fillna(0)
            df_transformed.columns = [f'{col} interest' if col != 'Year' else col for col in df_transformed.columns]
            df2 = df_transformed.reset_index()

            df3 = df_optimized_forecast

            pd.merge(df1, df3, on='Year', how='outer', suffixes=('_df1', '_df3'))
            calc_difference = [col for col in df1.columns if col != 'Year']
            forecast_dif_sold = pd.DataFrame({'Year': df1['Year']})
            forecast_dif_sold[calc_difference] = df3[calc_difference] - df1[calc_difference]

            pd.merge(df2, df3, on='Year', how='outer', suffixes=('_df2', '_df3'))
            calc_difference = [col for col in df2.columns if col != 'Year']
            forecast_dif_interest = pd.DataFrame({'Year': df2['Year']})
            forecast_dif_interest[calc_difference] = df3[calc_difference] - df2[calc_difference]

            title = '📈💰 Forecast performances savings'
            optimized_sold_chart(forecast_dif_sold, title, yaxis='Performance sold (€)')

            title = '📈🪙 Forecast performances interests'
            optimized_interest_chart(forecast_dif_interest, title, yaxis='Performance interest (€)')
