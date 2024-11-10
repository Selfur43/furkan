import pulp
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Parameters
operators = ['O_1', 'O_2', 'O_3', 'O_4', 'O_5']
machines = ['Machine_M_1', 'Machine_M_2', 'Machine_M_3']
shifts = ['Morning', 'Evening']
products = ['Metal_P_1', 'Metal_P_2', 'Metal_P_3']
workdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Dummy data for setup times and error rates (replace with actual data)
setup_times = {(i, j, k, p, d): np.random.randint(10, 30) for i in operators for j in machines for k in shifts for p in products for d in workdays}
error_rates = {(i, j, k, p, d): np.random.uniform(0.01, 0.1) for i in operators for j in machines for k in shifts for p in products for d in workdays}
skill_fit = {(i, j): np.random.uniform(0.8, 1.2) for i in operators for j in machines}

# Constraints and weights
max_daily_work_minutes = 600
daily_break_minutes = 60
weight_setup = 1.0
weight_error = 1.0
weight_shift_balance = 0.1
weight_workload_balance = 0.05
penalty_factor = 1.5

# Error rates'in dakikaya dönüştürülmesi
error_times = {key: error_rate * setup_times[key] for key, error_rate in error_rates.items()}

# Skill fit'in normalizasyonu
skill_min, skill_max = min(skill_fit.values()), max(skill_fit.values())
normalized_skill_fit = {key: (value - skill_min) / (skill_max - skill_min) for key, value in skill_fit.items()}


# Decision variables
x = pulp.LpVariable.dicts("x", (operators, machines, shifts, products, workdays), cat="Binary")
dev_shift_pos = pulp.LpVariable.dicts("dev_shift_pos", shifts, lowBound=0, cat="Integer")
dev_shift_neg = pulp.LpVariable.dicts("dev_shift_neg", shifts, lowBound=0, cat="Integer")
dev_workload_pos = pulp.LpVariable.dicts("dev_workload_pos", operators, lowBound=0, cat="Integer")
dev_workload_neg = pulp.LpVariable.dicts("dev_workload_neg", operators, lowBound=0, cat="Integer")

# Model definition
model = pulp.LpProblem("Operator_Assignment", pulp.LpMinimize)

# Objective function with balanced weights for minimizing setup times and error rates
model += (
    pulp.lpSum(
        (weight_setup * setup_times[i, j, k, p, d] + weight_error * error_rates[i, j, k, p, d] - (0.05 * normalized_skill_fit[i, j]* max_daily_work_minutes)) * x[i][j][k][p][d]
        for i in operators for j in machines for k in shifts for p in products for d in workdays
    )
    + weight_shift_balance * pulp.lpSum(dev_shift_pos[k] + dev_shift_neg[k] for k in shifts)
    + weight_workload_balance * pulp.lpSum(dev_workload_pos[i] + dev_workload_neg[i] for i in operators)
    + penalty_factor * pulp.lpSum(
        (error_rates[i, j, k, p, d] + setup_times[i, j, k, p, d]  - (0.05 * normalized_skill_fit[i, j]* max_daily_work_minutes)) * x[i][j][k][p][d]
        for i in operators for j in machines for k in shifts for p in products for d in workdays
    )
)

# Constraints
for i in operators:
    for d in workdays:
        model += pulp.lpSum(x[i][j][k][p][d] for j in machines for k in shifts for p in products) <= 1

for k in shifts:
    for d in workdays:
        model += pulp.lpSum(x[i][j][k][p][d] for i in operators for j in machines for p in products) >= 2
        model += pulp.lpSum(x[i][j][k][p][d] for i in operators for j in machines for p in products) <= 5

for i in operators:
    for d in range(len(workdays) - 1):
        for j in machines:
            model += pulp.lpSum(x[i][j]["Evening"][p][workdays[d]] for p in products) + \
                     pulp.lpSum(x[i][j]["Morning"][p][workdays[d + 1]] for p in products) <= 1

for i in operators:
    model += pulp.lpSum(x[i][j][k][p][d] for j in machines for k in shifts for p in products for d in workdays) <= 5

for k in shifts:
    model += (
        pulp.lpSum(x[i][j][k][p][d] for i in operators for j in machines for p in products for d in workdays) 
        == 3 + dev_shift_pos[k] - dev_shift_neg[k]
    )

for i in operators:
    model += (
        pulp.lpSum(setup_times[i, j, k, p, d] * x[i][j][k][p][d] for j in machines for k in shifts for p in products for d in workdays)
        == 360 + dev_workload_pos[i] - dev_workload_neg[i]
    )

# Solve the model
model.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=3600, gapRel=0.05))

# Check if a feasible solution is found
if model.status == pulp.LpStatusOptimal or model.status == pulp.LpStatusInfeasible:
    print(f"Solution Status: {pulp.LpStatus[model.status]}")
    results = []
    for i in operators:
        for j in machines:
            for k in shifts:
                for p in products:
                    for d in workdays:
                        if x[i][j][k][p][d].varValue == 1:
                            results.append([i, j, k, p, d, setup_times[i, j, k, p, d], error_rates[i, j, k, p, d], skill_fit[i, j]])

    # Display results in a DataFrame
    df_results = pd.DataFrame(results, columns=["Operator", "Machine", "Shift", "Product", "Day", "Setup Time (min)", "Error Rate (%)", "Skill Score"])
    st.dataframe(df_results)

# Visualizations in Streamlit
st.title("📊 Operator Assignment Optimization Results")
st.subheader("📋 Optimized Operator Assignment Table")

# Analytical Summary
st.subheader("📈 Analytical Summary")
total_setup_time = df_results["Setup Time (min)"].sum()
average_error_rate = df_results["Error Rate (%)"].mean()
st.metric("Total Setup Time (min)", f"{total_setup_time:.2f}")
st.metric("Average Error Rate (%)", f"{average_error_rate:.2%}")

# Operator Workload Distribution
st.subheader("⚙️ Operator Workload Distribution")
df_workload = df_results.groupby("Operator").agg(Total_Workload=pd.NamedAgg(column="Setup Time (min)", aggfunc="sum")).reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="Operator", y="Total_Workload", data=df_workload, ax=ax, palette="viridis")
ax.set_title("Operator Workload by Setup Time (min)")
ax.set_xlabel("Operator")
ax.set_ylabel("Total Setup Time (min)")
st.pyplot(fig)

# Setup Time Analysis by Shift and Machine
st.subheader("🔍 Setup Time Analysis by Shift and Machine")
df_setup_by_shift_machine = df_results.groupby(["Shift", "Machine"]).agg(Total_Setup_Time=pd.NamedAgg(column="Setup Time (min)", aggfunc="sum")).reset_index()
fig = px.sunburst(df_setup_by_shift_machine, path=["Shift", "Machine"], values="Total_Setup_Time", title="Total Setup Time Distribution by Shift and Machine")
st.plotly_chart(fig)

# Error Rate Analysis
st.subheader("🔍 Error Rate Distribution by Operator")
df_error_rate = df_results.groupby("Operator").agg(Average_Error_Rate=pd.NamedAgg(column="Error Rate (%)", aggfunc="mean")).reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="Operator", y="Average_Error_Rate", data=df_error_rate, ax=ax, palette="coolwarm")
ax.set_title("Average Error Rate by Operator")
ax.set_xlabel("Operator")
ax.set_ylabel("Average Error Rate (%)")
st.pyplot(fig)

