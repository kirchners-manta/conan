import pulp

# Define the initial parameters
T_initial = 576  # Total initial atoms in the undoped sheet
P_desired = 8.5  # Desired nitrogen percentage in the doped sheet
D = 5  # Number of doping types

# Constants for each doping type
ci = [0, 1, 1, 1, 2]  # Number of carbon atoms removed by doping type i
ri = [1, 1, 2, 3, 4]  # Number of nitrogen atoms added by doping type i
names = ["Graphitic-N", "Pyridinic-N1", "Pyridinic-N2", "Pyridinic-N3", "Pyridinic-N4"]

# Compute ki values
P_desired_fraction = P_desired / 100.0  # Convert desired percentage to fraction
# Effective nitrogen contribution of doping type i, accounting for nitrogen added and the effect of carbon atoms removed
# on the overall nitrogen percentage
ki = [ri_i + P_desired_fraction * ci_i for ri_i, ci_i in zip(ri, ci)]
# The right-hand side of the equation, representing the desired total nitrogen atoms based on the initial total atoms
RHS = P_desired_fraction * T_initial

# Initialize the problem
prob = pulp.LpProblem("Nitrogen_Doping_Optimization", pulp.LpMinimize)

# Decision variables
# Integer variable representing the number of doping structures of type i to insert
xi = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(D)]
# Continuous variable representing the average number of nitrogen atoms per doping type
N_avg = pulp.LpVariable("N_avg", lowBound=0, cat="Continuous")
# di = [pulp.LpVariable(f"d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
# Continuous variables for positive and negative deviations of nitrogen atoms added by doping type i from N_avg
P_di = [pulp.LpVariable(f"P_d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
N_di = [pulp.LpVariable(f"N_d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
# Continuous variables for positive and negative deviations (in nitrogen atom units) from the desired total nitrogen
# atoms
P = pulp.LpVariable("P_i", lowBound=0, cat="Continuous")
N = pulp.LpVariable("N_i", lowBound=0, cat="Continuous")
# Continuous variable representing the deviation in nitrogen percentage from the desired value
z1 = pulp.LpVariable("z1", lowBound=0, cat="Continuous")
# Continuous variable representing the total deviation in nitrogen atoms from equal distribution among doping types
z2 = pulp.LpVariable("z2", lowBound=0, cat="Continuous")

# Objective function
w1 = 1000  # Weight for deviation from desired nitrogen percentage
w2 = 1  # Weight for deviation from equal distribution of nitrogen atoms
prob += w1 * z1 + w2 * z2, "Minimize total deviation"

# # Constraints for the nitrogen percentage
# prob += pulp.lpSum([ki[i] * xi[i] for i in range(D)]) - RHS <= z1, "Upper bound constraint"
# prob += RHS - pulp.lpSum([ki[i] * xi[i] for i in range(D)]) <= z1, "Lower bound constraint"

# Nitrogen percentage constraint (replacing upper and lower bound constraints) to ensure that the total effective
# nitrogen contribution from all doping types matches the desired total nitrogen atoms (RHS), accounting for deviations
# (P, N)
prob += pulp.lpSum([ki[i] * xi[i] for i in range(D)]) + P - N == RHS, "Nitrogen deviation constraint"
# Define z1 as the sum of positive and negative deviations, effectively capturing the absolute deviation in nitrogen
# atoms
prob += z1 == P + N, "Absolute deviation constraint"

# Constraint to calculate the average number of nitrogen atoms added per doping type
prob += N_avg == pulp.lpSum([ri[i] * xi[i] for i in range(D)]) / D, "Average nitrogen atoms constraint"

# # Constraints for deviations in nitrogen atoms from the average
# for i in range(D):
#     prob += di[i] >= ri[i] * xi[i] - N_avg, f"Deviation_positive_{i}"
#     prob += di[i] >= N_avg - ri[i] * xi[i], f"Deviation_negative_{i}"
# prob += z2 == pulp.lpSum([di[i] for i in range(D)]), "Total deviation in nitrogen atoms"

# Constraints for deviations in nitrogen atoms from the average
# for i in range(D):
#     prob += ri[i] * xi[i] - N_avg == P_di[i] - N_di[i], f"Nitrogen deviation for type {i}"
for i in range(D):
    prob += ri[i] * xi[i] + P_di[i] - N_di[i] == N_avg, f"Nitrogen deviation for type {i}"

# Define z2 as the sum of all individual deviations, representing the total deviation from equal nitrogen distribution
prob += z2 == pulp.lpSum([P_di[i] + N_di[i] for i in range(D)]), "Total deviation in nitrogen atoms"

# Solve the problem
prob.solve(pulp.PULP_CBC_CMD(msg=True))

# Retrieve the solution (optimized values of the variables from the solved model)
xi_values = [int(xi[i].varValue) for i in range(D)]
N_avg_value = N_avg.varValue
P_value = P.varValue
N_value = N.varValue
# di_values = [di[i].varValue for i in range(D)]
P_di_values = [P_di[i].varValue for i in range(D)]
N_di_values = [N_di[i].varValue for i in range(D)]
z1_value = z1.varValue
z2_value = z2.varValue

# Compute the actual nitrogen percentage
N_total = sum([ri[i] * xi_values[i] for i in range(D)])  # Total  nitrogen atoms added
C_removed = sum([ci[i] * xi_values[i] for i in range(D)])  # Total carbon atoms removed
T_final = T_initial - C_removed  # Final total atoms
P_actual = (N_total / T_final) * 100  # Actual nitrogen percentage

# Convert the absolute deviation in the number of nitrogen atoms to a percentage deviation for easier interpretation
z1_percentage = z1_value / (T_final / 100)

# Print the results
print("Doping Results:")
print("|Nitrogen Species | Actual Percentage (%) | Nitrogen Atom Count | Doping Structure Count |")
for i in range(D):
    species = names[i]
    N_atom_count = ri[i] * xi_values[i]
    doping_structure_count = xi_values[i]
    actual_percentage = (N_atom_count / T_final) * 100
    print(f"{i:<6}{species:<15}{actual_percentage:>17.2f}{N_atom_count:>22}{doping_structure_count:>25}")
print(f"{'Total Doping':<21}{P_actual:>17.2f}{N_total:>22}{sum(xi_values):>25}")

# Output the final objective value with scaled z1
print(f"\nTotal Deviation (Objective Value): {prob.objective.value()}")
print(f" - z1 (Deviation in Nitrogen Atoms): {z1_value} atoms")
print(f" - z1 as Percentage Deviation: {z1_percentage:.4f}%")
print(f" - z2 (Deviation from Equal Distribution): {z2_value} atoms")


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# Compute actual percentages per species
actual_percentages = [(ri[i] * xi_values[i] / T_final) * 100 for i in range(D)]
# Compute actual atoms per species
actual_atoms_per_species = [ri[i] * xi_values[i] for i in range(D)]

# Create a figure with a specific size
plt.figure(figsize=(14, 10))

# Define the grid layout
gs = gridspec.GridSpec(2, 2)

# Top Left Plot: Nitrogen Percentage per Species
ax1 = plt.subplot(gs[0, 0])
ax1.bar(names, actual_percentages, color="skyblue", label="Actual Nitrogen Percentage per Species")
ax1.axhline(y=P_desired / D, color="r", linestyle="--", label="Desired Nitrogen Percentage per Species")
ax1.set_xlabel("Nitrogen Species", fontsize=12)
ax1.set_ylabel("Nitrogen Percentage (%)", fontsize=12)
ax1.set_title("Nitrogen Percentage per Species", fontsize=14)
ax1.legend()
ax1.grid(True)

# Top Right Plot: Nitrogen Atoms per Species
ax2 = plt.subplot(gs[0, 1])
ax2.bar(names, actual_atoms_per_species, color="lightgreen", label="Actual Nitrogen Atoms per Species")
ax2.axhline(y=N_avg_value, color="r", linestyle="--", label="Average Nitrogen Atoms per Species")
ax2.set_xlabel("Nitrogen Species", fontsize=12)
ax2.set_ylabel("Nitrogen Atom Count", fontsize=12)
ax2.set_title("Nitrogen Atoms per Species", fontsize=14)
ax2.legend()
ax2.grid(True)

# Bottom Center Plot: Objective Function Components
ax3 = plt.subplot(gs[1, :])
deviations_values = [z1_value, z2_value]
deviations_labels = ["z1: Nitrogen Atom Deviation", "z2: Equal Distribution Deviation"]
bars = ax3.bar(deviations_labels, deviations_values, color=["blue", "green"])
ax3.set_ylabel("Deviation Magnitude (Number of Nitrogen Atoms)", fontsize=12)
ax3.set_title("Objective Function Components", fontsize=14)
for bar, value in zip(bars, deviations_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2, height, f"{value:.2f}", ha="center", va="bottom", fontsize=12)
ax3.grid(True)

plt.tight_layout()
plt.show()


import pulp

# Define the initial parameters
T_initial = 576  # Total initial atoms in the undoped sheet
P_desired = 8.5  # Desired nitrogen percentage in the doped sheet
D = 5  # Number of doping types

# Constants for each doping type
ci = [0, 1, 1, 1, 2]  # Number of carbon atoms removed by doping type i
ri = [1, 1, 2, 3, 4]  # Number of nitrogen atoms added by doping type i
names = ["Graphitic-N", "Pyridinic-N1", "Pyridinic-N2", "Pyridinic-N3", "Pyridinic-N4"]

# Compute ki values
P_desired_fraction = P_desired / 100.0  # Convert desired percentage to fraction
ki = [ri_i + P_desired_fraction * ci_i for ri_i, ci_i in zip(ri, ci)]
RHS = P_desired_fraction * T_initial

# Initialize the problem
prob = pulp.LpProblem("Nitrogen_Doping_Optimization", pulp.LpMinimize)

# Decision variables
xi = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(D)]
N_avg = pulp.LpVariable("N_avg", lowBound=0, cat="Continuous")
P_di = [pulp.LpVariable(f"P_d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
N_di = [pulp.LpVariable(f"N_d_{i}", lowBound=0, cat="Continuous") for i in range(D)]
P = pulp.LpVariable("P_i", lowBound=0, cat="Continuous")
N = pulp.LpVariable("N_i", lowBound=0, cat="Continuous")
z1 = pulp.LpVariable("z1", lowBound=0, cat="Continuous")
z2 = pulp.LpVariable("z2", lowBound=0, cat="Continuous")

# New integer variable to enforce evenness
y = pulp.LpVariable("y", lowBound=0, cat="Integer")

# Objective function
w1 = 1000  # Weight for deviation from desired nitrogen percentage
w2 = 1  # Weight for deviation from equal distribution of nitrogen atoms
prob += w1 * z1 + w2 * z2, "Minimize total deviation"

# Nitrogen percentage constraint
prob += pulp.lpSum([ki[i] * xi[i] for i in range(D)]) + P - N == RHS, "Nitrogen deviation constraint"
prob += z1 == P + N, "Absolute deviation constraint"

# Average nitrogen atoms constraint
prob += N_avg == pulp.lpSum([ri[i] * xi[i] for i in range(D)]) / D, "Average nitrogen atoms constraint"

# Constraints for deviations in nitrogen atoms from the average
for i in range(D):
    prob += ri[i] * xi[i] + P_di[i] - N_di[i] == N_avg, f"Nitrogen deviation for type {i}"
# Total deviation in nitrogen atoms
prob += z2 == pulp.lpSum([P_di[i] + N_di[i] for i in range(D)]), "Total deviation in nitrogen atoms"

# New constraint to enforce evenness of total nitrogen atoms
prob += pulp.lpSum([ri[i] * xi[i] for i in range(D)]) - 2 * y == 0, "Evenness constraint"

# Solve the problem
prob.solve(pulp.PULP_CBC_CMD(msg=True))

# Retrieve the solution
xi_values = [int(xi[i].varValue) for i in range(D)]
N_avg_value = N_avg.varValue
P_value = P.varValue
N_value = N.varValue
P_di_values = [P_di[i].varValue for i in range(D)]
N_di_values = [N_di[i].varValue for i in range(D)]
z1_value = z1.varValue
z2_value = z2.varValue
y_value = y.varValue

# Compute the actual nitrogen percentage
N_total = sum([ri[i] * xi_values[i] for i in range(D)])  # Total nitrogen atoms added
C_removed = sum([ci[i] * xi_values[i] for i in range(D)])  # Total carbon atoms removed
T_final = T_initial - C_removed  # Final total atoms
P_actual = (N_total / T_final) * 100  # Actual nitrogen percentage

# Convert the absolute deviation in the number of nitrogen atoms to a percentage deviation
z1_percentage = z1_value / (T_final / 100)

# Check if N_total is even
is_even = N_total % 2 == 0

# Print the results
print("Doping Results:")
print("| Index | Nitrogen Species | Actual Percentage (%) | Nitrogen Atom Count | Doping Structure Count |")
for i in range(D):
    species = names[i]
    N_atom_count = ri[i] * xi_values[i]
    doping_structure_count = xi_values[i]
    actual_percentage = (N_atom_count / T_final) * 100
    print(f"{i:<6}{species:<17}{actual_percentage:>20.2f}{N_atom_count:>22}{doping_structure_count:>25}")
print(f"{'Total Doping':<23}{P_actual:>20.2f}{N_total:>22}{sum(xi_values):>25}")

# Output the final objective value with scaled z1
print(f"\nTotal Deviation (Objective Value): {prob.objective.value()}")
print(f" - z1 (Deviation in Nitrogen Atoms): {z1_value} atoms")
print(f" - z1 as Percentage Deviation: {z1_percentage:.4f}%")
print(f" - z2 (Deviation from Equal Distribution): {z2_value} atoms")
print(f" - N_total (Total Nitrogen Atoms Added): {N_total}")
print(f" - y (Auxiliary Variable): {y_value}")
print(f" - Is N_total even? {'Yes' if is_even else 'No'}")
