# import pulp
# import numpy as np
#
# # Define the initial parameters
# T_initial = 576  # Total initial atoms
# P_desired = 8.0  # Desired nitrogen percentage
# D = 5  # Number of doping types
#
# # Constants for each doping type
# ci = [0, 1, 1, 1, 2]  # Carbon atoms removed
# ri = [1, 1, 2, 3, 4]  # Nitrogen atoms added (replaced)
# names = ["Graphitic-N", "Pyridinic-N1", "Pyridinic-N2", "Pyridinic-N3", "Pyridinic-N4"]
#
# # Scale factors to convert to integers
# scale_factor = 100
# P_desired_fraction = P_desired / 100.0
#
# # Calculate ki and RHS
# # ki values (coefficients) are weights that represent the change in the nitrogen and carbon content through the
# # application of a specific doping type i
# ki = [scale_factor * ri_i + scale_factor * P_desired_fraction * ci_i for ri_i, ci_i in zip(ri, ci)]
# # RHS is the right-hand side of the equation, representing the desired nitrogen content related to the overall
# structure
# RHS = P_desired_fraction * T_initial * scale_factor
#
# # Initialize the problem
# prob = pulp.LpProblem("Nitrogen_Doping_Optimization", pulp.LpMinimize)
#
# # Decision variables
# xi = [pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(D)]
# z = pulp.LpVariable("z", lowBound=0, cat='Continuous')
#
# # Objective function
# prob += z, "Minimize absolute difference"
#
# # Constraints for the absolute value linearization
# prob += pulp.lpSum([ki[i] * xi[i] for i in range(D)]) - RHS <= z, "Upper bound constraint"
# prob += RHS - pulp.lpSum([ki[i] * xi[i] for i in range(D)]) <= z, "Lower bound constraint"
#
# # Solve the problem
# prob.solve()
#
# # Retrieve the solution
# xi_values = [int(xi[i].varValue) for i in range(D)]
# z_value = pulp.value(prob.objective)
#
# # Compute the actual nitrogen percentage
# N_total = sum([ri[i] * xi_values[i] for i in range(D)])
# C_removed = sum([ci[i] * xi_values[i] for i in range(D)])
# T_final = T_initial - C_removed
# P_actual = (N_total / T_final) * 100
#
# # Compute nitrogen distribution variance
# N_contributions = [ri[i] * xi_values[i] for i in range(D)]
# mean_N = np.mean(N_contributions)
# variance_N = np.var(N_contributions)
#
# # Print the results
# print("Doping Results:")
# print("|Nitrogen Species | Actual Percentage | Nitrogen Atom Count | Doping Structure Count |")
# for i in range(D):
#     species = names[i]
#     N_atom_count = N_contributions[i]
#     doping_structure_count = xi_values[i]
#     actual_percentage = (N_atom_count / T_final) * 100
#     print(f"{i:<6}{species:<15}{actual_percentage:>17.2f}{N_atom_count:>22}{doping_structure_count:>25}")
# print(f"{'Total Doping':<21}{P_actual:>17.2f}{N_total:>22}{sum(xi_values):>25}")


# import pulp
#
# # Define the initial parameters
# T_initial = 576  # Total initial atoms
# P_desired = 8.0  # Desired nitrogen percentage
# D = 5  # Number of doping types
#
# # Constants for each doping type
# ci = [0, 1, 1, 1, 2]  # Carbon atoms removed
# ri = [1, 1, 2, 3, 4]  # Nitrogen atoms added (replaced)
# names = ["Graphitic-N", "Pyridinic-N1", "Pyridinic-N2", "Pyridinic-N3", "Pyridinic-N4"]
#
# # Compute ki values
# P_desired_fraction = P_desired / 100.0
# ki = [ri_i + P_desired_fraction * ci_i for ri_i, ci_i in zip(ri, ci)]
# RHS = P_desired_fraction * T_initial
#
# # Initialize the problem
# prob = pulp.LpProblem("Nitrogen_Doping_Optimization", pulp.LpMinimize)
#
# # Decision variables
# xi = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(D)]
# x_avg = pulp.LpVariable("x_avg", lowBound=0, cat="Continuous")
# di = [pulp.LpVariable(f"d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
# z1 = pulp.LpVariable("z1", lowBound=0, cat="Continuous")
# z2 = pulp.LpVariable("z2", lowBound=0, cat="Continuous")
#
# # Objective function
# w1 = 1  # Weight for deviation from desired nitrogen percentage
# w2 = 1000  # Weight for deviation from equal distribution
# prob += w1 * z1 + w2 * z2, "Minimize total deviation"
#
# # Constraints for the nitrogen percentage
# prob += pulp.lpSum([ki[i] * xi[i] for i in range(D)]) - RHS <= z1, "Upper bound constraint"
# prob += RHS - pulp.lpSum([ki[i] * xi[i] for i in range(D)]) <= z1, "Lower bound constraint"
#
# # Constraint for average calculation
# prob += x_avg * D == pulp.lpSum([xi[i] for i in range(D)]), "Average constraint"
#
# # Constraints for deviations from average
# for i in range(D):
#     prob += di[i] >= xi[i] - x_avg, f"Deviation_positive_{i}"
#     prob += di[i] >= x_avg - xi[i], f"Deviation_negative_{i}"
# prob += z2 == pulp.lpSum([di[i] for i in range(D)]), "Total deviation"
#
# # Solve the problem
# prob.solve()
#
# # Retrieve the solution
# xi_values = [int(xi[i].varValue) for i in range(D)]
# x_avg_value = x_avg.varValue
# di_values = [di[i].varValue for i in range(D)]
# z1_value = z1.varValue
# z2_value = z2.varValue
#
# # Compute the actual nitrogen percentage
# N_total = sum([ri[i] * xi_values[i] for i in range(D)])
# C_removed = sum([ci[i] * xi_values[i] for i in range(D)])
# T_final = T_initial - C_removed
# P_actual = (N_total / T_final) * 100
#
# # Print the results
# print("Doping Results:")
# print("|Nitrogen Species | Actual Percentage | Nitrogen Atom Count | Doping Structure Count |")
# for i in range(D):
#     species = names[i]
#     N_atom_count = ri[i] * xi_values[i]
#     doping_structure_count = xi_values[i]
#     actual_percentage = (N_atom_count / T_final) * 100
#     print(f"{i:<6}{species:<15}{actual_percentage:>17.2f}{N_atom_count:>22}{doping_structure_count:>25}")
# print(f"{'Total Doping':<21}{P_actual:>17.2f}{N_total:>22}{sum(xi_values):>25}")

# import pulp
#
# # Define the initial parameters
# T_initial = 576  # Total initial atoms
# P_desired = 8.0  # Desired nitrogen percentage
# D = 5  # Number of doping types
#
# # Constants for each doping type
# ci = [0, 1, 1, 1, 2]  # Carbon atoms removed
# ri = [1, 1, 2, 3, 4]  # Nitrogen atoms added (replaced)
# names = ["Graphitic-N", "Pyridinic-N1", "Pyridinic-N2", "Pyridinic-N3", "Pyridinic-N4"]
#
# # Compute ki values
# P_desired_fraction = P_desired / 100.0
# ki = [ri_i + P_desired_fraction * ci_i for ri_i, ci_i in zip(ri, ci)]
# RHS = P_desired_fraction * T_initial
#
# # Initialize the problem
# prob = pulp.LpProblem("Nitrogen_Doping_Optimization", pulp.LpMinimize)
#
# # Decision variables
# xi = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(D)]
# N_avg = pulp.LpVariable("N_avg", lowBound=0, cat="Continuous")
# # di = [pulp.LpVariable(f"d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
# P_di = [pulp.LpVariable(f"P_d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
# N_di = [pulp.LpVariable(f"N_d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
# P = pulp.LpVariable("P_i", lowBound=0, cat="Continuous")
# N = pulp.LpVariable("N_i", lowBound=0, cat="Continuous")
# z1 = pulp.LpVariable("z1", lowBound=0, cat="Continuous")
# z2 = pulp.LpVariable("z2", lowBound=0, cat="Continuous")
#
# # Objective function
# w1 = 1  # Weight for deviation from desired nitrogen percentage
# w2 = 1  # Weight for deviation from equal distribution of nitrogen atoms
# prob += w1 * z1 + w2 * z2, "Minimize total deviation"
#
# # # Constraints for the nitrogen percentage
# # prob += pulp.lpSum([ki[i] * xi[i] for i in range(D)]) - RHS <= z1, "Upper bound constraint"
# # prob += RHS - pulp.lpSum([ki[i] * xi[i] for i in range(D)]) <= z1, "Lower bound constraint"
#
# # Nitrogen percentage constraint (replacing upper and lower bound constraints)
# prob += pulp.lpSum([ki[i] * xi[i] for i in range(D)]) + P - N == RHS, "Nitrogen deviation constraint"
# prob += z1 == P + N, "Absolute deviation constraint"
#
# # Constraint for average nitrogen atoms calculation
# prob += N_avg * D == pulp.lpSum([ri[i] * xi[i] for i in range(D)]), "Average nitrogen atoms constraint"
#
# # # Constraints for deviations in nitrogen atoms from the average
# # for i in range(D):
# #     prob += di[i] >= ri[i] * xi[i] - N_avg, f"Deviation_positive_{i}"
# #     prob += di[i] >= N_avg - ri[i] * xi[i], f"Deviation_negative_{i}"
# # prob += z2 == pulp.lpSum([di[i] for i in range(D)]), "Total deviation in nitrogen atoms"
#
# # Constraints for deviations in nitrogen atoms from the average without di
# # for i in range(D):
# #     prob += ri[i] * xi[i] - N_avg == P_di[i] - N_di[i], f"Nitrogen deviation for type {i}"
# for i in range(D):
#     prob += ri[i] * xi[i] + P_di[i] - N_di[i] == N_avg, f"Nitrogen deviation for type {i}"
#
# # Total deviation in nitrogen atoms (z2) as the sum of P_di and N_di
# prob += z2 == pulp.lpSum([P_di[i] + N_di[i] for i in range(D)]), "Total deviation in nitrogen atoms"
#
# # Solve the problem
# prob.solve()
#
# # Retrieve the solution
# xi_values = [int(xi[i].varValue) for i in range(D)]
# N_avg_value = N_avg.varValue
# P_value = P.varValue
# N_value = N.varValue
# # di_values = [di[i].varValue for i in range(D)]
# P_di_values = [P_di[i].varValue for i in range(D)]
# N_di_values = [N_di[i].varValue for i in range(D)]
# z1_value = z1.varValue
# z2_value = z2.varValue
#
# # Compute the actual nitrogen percentage
# N_total = sum([ri[i] * xi_values[i] for i in range(D)])
# C_removed = sum([ci[i] * xi_values[i] for i in range(D)])
# T_final = T_initial - C_removed
# P_actual = (N_total / T_final) * 100
#
# # Scale the actual deviation in nitrogen percentage (z1) with T_final
# z1_percentage = z1_value / (T_final / 100)
#
# # Print the results
# print("Doping Results:")
# print("|Nitrogen Species | Actual Percentage | Nitrogen Atom Count | Doping Structure Count |")
# for i in range(D):
#     species = names[i]
#     N_atom_count = ri[i] * xi_values[i]
#     doping_structure_count = xi_values[i]
#     actual_percentage = (N_atom_count / T_final) * 100
#     print(f"{i:<6}{species:<15}{actual_percentage:>17.2f}{N_atom_count:>22}{doping_structure_count:>25}")
# print(f"{'Total Doping':<21}{P_actual:>17.2f}{N_total:>22}{sum(xi_values):>25}")
#
# # Output the final objective value with scaled z1
# print(f"Objective value (z1 + z2): {prob.objective.value()}")
# print(f"Actual deviation in nitrogen percentage (scaled z1): {z1_percentage}")
# print("\nDone")

import pulp

# Define the initial parameters
T_initial = 576  # Total initial atoms
P_desired = 8.0  # Desired nitrogen percentage
D = 5  # Number of doping types

# Constants for each doping type
ci = [0, 1, 1, 1, 2]  # Carbon atoms removed
ri = [1, 1, 2, 3, 4]  # Nitrogen atoms added (replaced)
names = ["Graphitic-N", "Pyridinic-N1", "Pyridinic-N2", "Pyridinic-N3", "Pyridinic-N4"]

# Compute ki values
P_desired_fraction = P_desired / 100.0
target_nitrogen = P_desired_fraction * T_initial  # Total target nitrogen atoms
ki = [ri_i + P_desired_fraction * ci_i for ri_i, ci_i in zip(ri, ci)]

# Initialize the optimization problem
prob = pulp.LpProblem("Nitrogen_Doping_Optimization", pulp.LpMinimize)

# Decision variables
xi = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(D)]
N_avg = pulp.LpVariable("N_avg", lowBound=0, cat="Continuous")
P = pulp.LpVariable("P_i", lowBound=0, cat="Continuous")
N = pulp.LpVariable("N_i", lowBound=0, cat="Continuous")
z1 = pulp.LpVariable("z1", lowBound=0, cat="Continuous")
z2 = pulp.LpVariable("z2", lowBound=0, cat="Continuous")

# Deviation variables for individual species
P_di = [pulp.LpVariable(f"P_d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
N_di = [pulp.LpVariable(f"N_d_{i}", lowBound=0, cat="Continuous") for i in range(D)]

# Objective function with scaling factors for normalization
scaling_factor_z1 = 1 / P_desired
scaling_factor_z2 = 1 / target_nitrogen
w1 = 1000  # Weight for nitrogen percentage deviation
w2 = 1  # Weight for nitrogen distribution deviation
prob += w1 * scaling_factor_z1 * z1 + w2 * scaling_factor_z2 * z2, "Minimize normalized total deviation"

# Nitrogen percentage constraint (deviation from target percentage)
prob += pulp.lpSum([ki[i] * xi[i] for i in range(D)]) + P - N == target_nitrogen, "Nitrogen deviation constraint"
prob += z1 == P + N, "Absolute deviation constraint for percentage"

# Constraint for average nitrogen atoms calculation
prob += N_avg * D == pulp.lpSum([ri[i] * xi[i] for i in range(D)]), "Average nitrogen atoms constraint"

# Constraints for deviations in nitrogen atoms from the average
for i in range(D):
    prob += ri[i] * xi[i] + P_di[i] - N_di[i] == N_avg, f"Nitrogen deviation for type {i}"

# Total deviation in nitrogen atoms (z2) as the sum of P_di and N_di
prob += z2 == pulp.lpSum([P_di[i] + N_di[i] for i in range(D)]), "Total deviation in nitrogen atoms"

# Solve the problem
prob.solve()

# Retrieve the solution
xi_values = [int(xi[i].varValue) for i in range(D)]
N_avg_value = N_avg.varValue
P_value = P.varValue
N_value = N.varValue
P_di_values = [P_di[i].varValue for i in range(D)]
N_di_values = [N_di[i].varValue for i in range(D)]
z1_value = z1.varValue
z2_value = z2.varValue

# Compute the actual nitrogen percentage after calculating T_final
N_total = sum([ri[i] * xi_values[i] for i in range(D)])
C_removed = sum([ci[i] * xi_values[i] for i in range(D)])
T_final = T_initial - C_removed
P_actual = (N_total / T_final) * 100

# Scale the actual deviation in nitrogen percentage (z1) with T_final
z1_percentage = z1_value / (T_final / 100)

# Print the results
print("Doping Results:")
print("|Nitrogen Species | Actual Percentage | Nitrogen Atom Count | Doping Structure Count |")
for i in range(D):
    species = names[i]
    N_atom_count = ri[i] * xi_values[i]
    doping_structure_count = xi_values[i]
    actual_percentage = (N_atom_count / T_final) * 100
    print(f"{i:<6}{species:<15}{actual_percentage:>17.2f}{N_atom_count:>22}{doping_structure_count:>25}")
print(f"{'Total Doping':<21}{P_actual:>17.2f}{N_total:>22}{sum(xi_values):>25}")

# Output the final objective value with normalized z1 and z2
print(f"Objective value (normalized z1 + z2): {prob.objective.value()}")
print(f"Actual deviation in nitrogen percentage (scaled z1): {z1_percentage}")
print("\nDone")


# import matplotlib.pyplot as plt
#
# # Species names and calculated nitrogen percentages based on optimized results
# species = ["Graphitic-N", "Pyridinic-N1", "Pyridinic-N2", "Pyridinic-N3", "Pyridinic-N4"]
# actual_percentages = [(ri[i] * xi_values[i] / T_final) * 100 for i in range(D)]
# desired_percentage_per_species = [P_desired / D] * len(species)  # Equal target distribution
#
# # Create the plot
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # Plot desired nitrogen percentage per species as black circles
# ax.plot(species, desired_percentage_per_species, "ko", label="Desired Nitrogen Percentage per Species")
#
# # Plot the actual nitrogen percentage as a continuous blue line
# ax.plot(species, actual_percentages, "b", label="Actual Nitrogen Percentage per Species")
#
# # Annotate deviations with correct "N" or "P" labeling based on relative position of actual and desired values
# for i, (act, des) in enumerate(zip(actual_percentages, desired_percentage_per_species)):
#     if des < act:  # Actual above desired (positive deviation), should be "P"
#         ax.plot([species[i], species[i]], [des, act], linestyle="--", color="orange")  # Orange dashed line
#         ax.text(i, (des + act) / 2, f"P_{i+1}", ha="right", va="top", fontsize=12, color="orange")
#     elif des > act:  # Desired above actual (negative deviation), should be "N"
#         ax.plot([species[i], species[i]], [act, des], linestyle="--", color="orange")  # Orange dashed line
#         ax.text(i, (des + act) / 2, f"N_{i+1}", ha="right", va="bottom", fontsize=12, color="orange")
#
# # Axis labels and title
# ax.set_xlabel("Nitrogen Species", fontsize=14)
# ax.set_ylabel("Nitrogen Percentage (%)", fontsize=14)
# ax.set_title("Desired vs. Actual Nitrogen Percentage per Species with Correct Deviations (P and N)", fontsize=16)
# ax.legend()
#
# # Show grid and plot
# plt.grid(True)
# plt.show()
# print("\nDone")

import matplotlib.pyplot as plt
import numpy as np

# Visualization of objective functions

# Objective Function 1: Nitrogen Percentage Deviation
species = ["Graphitic-N", "Pyridinic-N1", "Pyridinic-N2", "Pyridinic-N3", "Pyridinic-N4"]
actual_percentages = [(ri[i] * xi_values[i] / T_final) * 100 for i in range(D)]
desired_percentage_per_species = [P_desired / D] * len(species)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(species, desired_percentage_per_species, "ko", label="Desired Nitrogen Percentage per Species")
ax1.plot(species, actual_percentages, "b", label="Actual Nitrogen Percentage per Species")
for i, (act, des) in enumerate(zip(actual_percentages, desired_percentage_per_species)):
    if des < act:
        ax1.plot([species[i], species[i]], [des, act], linestyle="--", color="orange")
        ax1.text(i, (des + act) / 2, f"P_{i+1}", ha="right", va="top", fontsize=12, color="orange")
    elif des > act:
        ax1.plot([species[i], species[i]], [act, des], linestyle="--", color="orange")
        ax1.text(i, (des + act) / 2, f"N_{i+1}", ha="right", va="bottom", fontsize=12, color="orange")
ax1.set_xlabel("Nitrogen Species", fontsize=14)
ax1.set_ylabel("Nitrogen Percentage (%)", fontsize=14)
ax1.set_title("Objective Function 1: Nitrogen Percentage Deviation", fontsize=16)
ax1.legend()
plt.grid(True)

# Objective Function 2: Deviation from Equal Nitrogen Distribution
fig, ax2 = plt.subplots(figsize=(10, 6))
N_avg_values = [N_avg_value] * len(species)
actual_atoms_per_species = [ri[i] * xi_values[i] for i in range(D)]
ax2.plot(species, N_avg_values, "ko", label="Average Nitrogen Atoms per Species")
ax2.plot(species, actual_atoms_per_species, "b", label="Actual Nitrogen Atoms per Species")
for i, (actual, avg) in enumerate(zip(actual_atoms_per_species, N_avg_values)):
    if actual > avg:
        ax2.plot([species[i], species[i]], [avg, actual], linestyle="--", color="orange")
        ax2.text(i, (avg + actual) / 2, f"P_d_{i+1}", ha="right", va="top", fontsize=12, color="orange")
    elif actual < avg:
        ax2.plot([species[i], species[i]], [actual, avg], linestyle="--", color="orange")
        ax2.text(i, (avg + actual) / 2, f"N_d_{i+1}", ha="right", va="bottom", fontsize=12, color="orange")
ax2.set_xlabel("Nitrogen Species", fontsize=14)
ax2.set_ylabel("Nitrogen Atom Count", fontsize=14)
ax2.set_title("Objective Function 2: Deviation from Equal Nitrogen Distribution", fontsize=16)
ax2.legend()
plt.grid(True)

# Combined Objective Function: Total Deviation
fig, ax3 = plt.subplots(figsize=(10, 6))
total_deviation_z1 = np.sum([abs(desired_percentage_per_species[i] - actual_percentages[i]) for i in range(D)])
total_deviation_z2 = np.sum([abs(N_avg_values[i] - actual_atoms_per_species[i]) for i in range(D)])
ax3.bar(["z1", "z2"], [total_deviation_z1, total_deviation_z2], color=["blue", "green"])
ax3.set_ylabel("Deviation Magnitude", fontsize=14)
ax3.set_title("Combined Objective Function: Sum of Deviations", fontsize=16)
plt.grid(True)

plt.show()
