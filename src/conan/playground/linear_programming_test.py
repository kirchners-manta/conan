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
ki = [ri_i + P_desired_fraction * ci_i for ri_i, ci_i in zip(ri, ci)]
RHS = P_desired_fraction * T_initial

# Initialize the problem
prob = pulp.LpProblem("Nitrogen_Doping_Optimization", pulp.LpMinimize)

# Decision variables
xi = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(D)]
x_avg = pulp.LpVariable("x_avg", lowBound=0, cat="Continuous")
di = [pulp.LpVariable(f"d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
z1 = pulp.LpVariable("z1", lowBound=0, cat="Continuous")
z2 = pulp.LpVariable("z2", lowBound=0, cat="Continuous")

# Objective function
w1 = 1000  # Weight for deviation from desired nitrogen percentage
w2 = 1  # Weight for deviation from equal distribution
prob += w1 * z1 + w2 * z2, "Minimize total deviation"

# Constraints for the nitrogen percentage
prob += pulp.lpSum([ki[i] * xi[i] for i in range(D)]) - RHS <= z1, "Upper bound constraint"
prob += RHS - pulp.lpSum([ki[i] * xi[i] for i in range(D)]) <= z1, "Lower bound constraint"

# Constraint for average calculation
prob += x_avg * D == pulp.lpSum([xi[i] for i in range(D)]), "Average constraint"

# Constraints for deviations from average
for i in range(D):
    prob += di[i] >= xi[i] - x_avg, f"Deviation_positive_{i}"
    prob += di[i] >= x_avg - xi[i], f"Deviation_negative_{i}"
prob += z2 == pulp.lpSum([di[i] for i in range(D)]), "Total deviation"

# Solve the problem
prob.solve()

# Retrieve the solution
xi_values = [int(xi[i].varValue) for i in range(D)]
x_avg_value = x_avg.varValue
di_values = [di[i].varValue for i in range(D)]
z1_value = z1.varValue
z2_value = z2.varValue

# Compute the actual nitrogen percentage
N_total = sum([ri[i] * xi_values[i] for i in range(D)])
C_removed = sum([ci[i] * xi_values[i] for i in range(D)])
T_final = T_initial - C_removed
P_actual = (N_total / T_final) * 100

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
