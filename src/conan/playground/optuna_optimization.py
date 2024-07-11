import optuna
from doping_experiment import Graphene, NitrogenSpecies, write_xyz


def objective(trial):
    # Sample k_inner and k_outer
    k_inner = trial.suggest_float("k_inner", 1.0, 1000.0, log=True)
    k_outer = trial.suggest_float("k_outer", 0.01, 10.0, log=True)

    # Create Graphene instance and set k_inner and k_outer
    # ToDo: Später soll es auch möglich sein, innerhalb für Winkel und Bindungen unterschiedliche Werte zu setzen
    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))
    graphene.k_inner = k_inner
    graphene.k_outer = k_outer

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3})

    # Calculate the total energy of the graphene sheet
    total_energy = graphene.calculate_total_energy()

    # Calculate bond and angle accuracy within cycles (additional objectives can be added)
    bond_accuracy, angle_accuracy = graphene.calculate_bond_angle_accuracy()

    # Combine objectives
    objective_value = total_energy + bond_accuracy + angle_accuracy

    # Optional: Schreibe die optimierte Struktur in eine Datei
    write_xyz(
        graphene.graph,
        f"optimized_graphene_k_inner_{k_inner}_k_outer_{k_outer}.xyz",
    )

    return objective_value


# Create Optuna study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Print best parameters
print("Best trial:")
trial = study.best_trial
print(f'k_inner: {trial.params["k_inner"]}')
print(f'k_outer: {trial.params["k_outer"]}')
