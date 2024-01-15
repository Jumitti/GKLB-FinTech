import os
import subprocess
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
app_script_path = os.path.join(current_directory, "GKLB_FinTech.py")
requirements_path = os.path.join(current_directory, "requirements.txt")

start_messages = [
    "------------------------------------------------------------------------------------------------------",
    "!                                          Welcome to GK!LB                                          !",
    "------------------------------------------------------------------------------------------------------",
    "GK!LB is a comprehensive financial tool designed to monitor the performance of savings accounts across",
    "various depositories. With a user-friendly interface, users can seamlessly optimize their investments",
    "by efficiently managing multiple savings accounts within a single application. The left sidebar",
    "facilitates easy addition of depositories and provides convenient controls for configuring savings",
    "accounts—enabling users to add, modify, or delete accounts with ease. GK!LB simplifies the process of",
    "tracking and optimizing savings, offering a streamlined and efficient financial management solution",
    "------------------------------------------------------------------------------------------------------",
    "Created by Minniti Julien - GitHub(https://github.com/Jumitti/GKLB-FinTech)",
    "MIT licence(https://github.com/Jumitti/GKLB-FinTech/blob/master/LICENSE)"
    ]
for message in start_messages:
    os.system(f"echo {message}")

time.sleep(2)

update_messages = "Installing/updating python prerequisites..."
os.system(f"echo {update_messages}")
subprocess.run(["pip", "install", "-r", requirements_path])

app_messages = "GK!LB Streamlit app running..."
subprocess.run(["streamlit", "run", app_script_path])
