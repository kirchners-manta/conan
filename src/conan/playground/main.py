import random

from conan.playground.doping import NitrogenSpecies, OptimizationWeights

# from conan.playground.labeling import AtomLabeler
from conan.playground.structures import CNT, GrapheneSheet, Pore
from conan.playground.utils import write_xyz

# from conan.playground.doping import NitrogenSpecies


# from conan.playground.doping import NitrogenSpecies


def main():
    # Set seed for reproducibility
    # random.seed(42)
    # random.seed(3)
    random.seed(1)

    ####################################################################################################################
    # # CREATE A GRAPHENE SHEET
    # sheet_size = (10, 10)
    #
    # graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    # write_xyz(graphene.graph, "graphene_sheet.xyz")

    ####################################################################################################################
    # # CREATE A GRAPHENE SHEET AND LABEL THE ATOMS
    # sheet_size = (10, 10)
    #
    # graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Label atoms before writing to XYZ file
    # labeler = AtomLabeler(graphene.graph, graphene.doping_handler.doping_structures)
    # labeler.label_atoms()
    #
    # write_xyz(graphene.graph, "graphene_sheet.xyz")

    ####################################################################################################################
    # CREATE A GRAPHENE SHEET, DOPE IT AND ADJUST POSITIONS VIA ADD_NITROGEN_DOPING METHOD
    sheet_size = (40, 40)

    # # Use default optimization weights
    # weights = OptimizationWeights()

    # Define optimization weights
    weights = OptimizationWeights(
        nitrogen_percentage_weight=1000,
        equal_distribution_weight=1,
    )

    graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
    # graphene.add_nitrogen_doping(total_percentage=8, adjust_positions=False, optimization_weights=weights)
    # graphene.add_nitrogen_doping(optimization_weights=weights)
    graphene.add_nitrogen_doping(
        total_percentage=10,
        percentages={NitrogenSpecies.PYRIDINIC_4: 3, NitrogenSpecies.PYRIDINIC_2: 2},
        optimization_weights=weights,
    )
    # graphene.add_nitrogen_doping(
    #     percentages={
    #         NitrogenSpecies.GRAPHITIC: 0.73,
    #         NitrogenSpecies.PYRIDINIC_1: 2.73,
    #         NitrogenSpecies.PYRIDINIC_2: 1.45,
    #         NitrogenSpecies.PYRIDINIC_3: 1.64,
    #         NitrogenSpecies.PYRIDINIC_4: 1.45,
    #     }, adjust_positions=False
    # )
    # # graphene.add_nitrogen_doping(total_percentage=10,
    # #                              percentages={NitrogenSpecies.PYRIDINIC_4: 2, NitrogenSpecies.GRAPHITIC: 3})
    graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)

    write_xyz(graphene.graph, "graphene_sheet_doped.xyz")

    ####################################################################################################################
    # # CREATE A GRAPHENE SHEET, DOPE IT AND ADJUST POSITIONS
    # sheet_size = (20, 20)
    #
    # # Create a graphene sheet
    # graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
    #
    # # Add nitrogen doping without adjusting positions
    # graphene.add_nitrogen_doping(total_percentage=10, adjust_positions=False)
    # # graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 1})
    #
    # # Adjust positions separately
    # graphene.adjust_atom_positions()
    # # Positions are now adjusted
    #
    # # Attempt to adjust positions again
    # graphene.adjust_atom_positions()
    # # Warning: Positions have already been adjusted
    #
    # # Plot structure
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # write_xyz(graphene.graph, "graphene_sheet_doped.xyz")

    ####################################################################################################################
    # # CREATE A GRAPHENE SHEET, DOPE IT AND LABEL THE ATOMS
    # sheet_size = (20, 20)
    #
    # graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
    # graphene.add_nitrogen_doping(total_percentage=10, adjust_positions=False)
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Label atoms before writing to XYZ file
    # labeler = AtomLabeler(graphene.graph, graphene.doping_handler.doping_structures)
    # labeler.label_atoms()
    #
    # write_xyz(graphene.graph, "graphene_sheet_doped.xyz")

    ####################################################################################################################
    # # VERSION 1: CREATE A GRAPHENE SHEET, DOPE AND STACK IT
    # import time
    # sheet_size = (20, 20)
    #
    # # Create a graphene sheet
    # graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
    #
    # # Add nitrogen doping to the graphene sheet
    # start_time = time.time()  # Time the nitrogen doping process
    # graphene.add_nitrogen_doping(total_percentage=15, adjust_positions=True)
    # end_time = time.time()
    #
    # # Calculate the elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Time taken for nitrogen doping for a sheet of size {sheet_size}: {elapsed_time:.2f} seconds")
    #
    # # Plot the graphene sheet with nitrogen doping
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Stack the graphene sheet
    # stacked_graphene = graphene.stack(interlayer_spacing=3.35, number_of_layers=5)
    #
    # # Plot the stacked structure
    # stacked_graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Save the structure to a .xyz file
    # write_xyz(stacked_graphene.graph, "ABA_stacking.xyz")

    ####################################################################################################################
    # # VERSION 2: DIRECTLY USE THE STACKED GRAPHENE SHEET AND ADJUST POSITIONS VIA ADD_NITROGEN_DOPING METHOD
    #
    # # Create a graphene sheet
    # graphene_sheet = GrapheneSheet(bond_length=1.42, sheet_size=(40, 40))
    #
    # # Create stacked graphene using the graphene sheet
    # stacked_graphene = StackedGraphene(graphene_sheet, number_of_layers=5, stacking_type="ABA")
    #
    # # Add nitrogen doping to the specified graphene sheets
    # # stacked_graphene.add_nitrogen_doping(total_percentage=8, adjust_positions=True, layers="all")
    #
    # stacked_graphene.add_nitrogen_doping(
    #     percentages={
    #         NitrogenSpecies.GRAPHITIC: 0.73,
    #         NitrogenSpecies.PYRIDINIC_1: 2.6,  # ToDo: Eigentlich müsste hier 2.73 stehen, um auf 2.73 zu kommen???
    #         NitrogenSpecies.PYRIDINIC_2: 1.45,
    #         NitrogenSpecies.PYRIDINIC_3: 1.64,
    #         NitrogenSpecies.PYRIDINIC_4: 1.45,
    #     },
    #     adjust_positions=True,
    #     layers="all",
    # )
    #
    # # Plot the stacked structure
    # stacked_graphene.plot_structure(with_labels=False, visualize_periodic_bonds=False)
    #
    # write_xyz(stacked_graphene.graph, "ABA_stacking.xyz")

    ####################################################################################################################
    # # VERSION 2: DIRECTLY USE THE STACKED GRAPHENE SHEET AND ADJUST POSITIONS OF SPECIFIC LAYERS
    #
    # # Create a base graphene sheet
    # base_graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    #
    # # Create a stacked graphene structure
    # stacked_graphene = StackedGraphene(base_graphene, number_of_layers=3)
    #
    # # Add nitrogen doping to layers 0 and 1 without adjusting positions
    # stacked_graphene.add_nitrogen_doping(total_percentage=10, adjust_positions=False, layers=[0, 1])
    # # No positions adjusted
    #
    # # Adjust positions for layers 0 and 1
    # stacked_graphene.adjust_atom_positions(layers=[0, 1])
    # # Positions are now adjusted for layers 0 and 1
    #
    # # Attempt to adjust positions again
    # stacked_graphene.adjust_atom_positions(layers=[0, 1])
    # # Warnings: Positions have already been adjusted in layers 0 and 1

    ####################################################################################################################
    # # Example: Only dope the first and last layer (both will have the same doping percentage but different ordering)
    # import time
    #
    # sheet_size = (20, 20)
    #
    # # Create a graphene sheet
    # graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
    #
    # # Stack the graphene sheet
    # stacked_graphene = graphene.stack(interlayer_spacing=3.34, number_of_layers=5, stacking_type="ABC")
    #
    # # Add individual nitrogen doping only to the first and last layer
    # start_time = time.time()  # Time the nitrogen doping process
    # stacked_graphene.add_nitrogen_doping_to_layer(layer_index=0, total_percentage=15, adjust_positions=True)
    # stacked_graphene.add_nitrogen_doping_to_layer(layer_index=4, total_percentage=15, adjust_positions=True)
    # end_time = time.time()
    #
    # # Calculate the elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Time taken for nitrogen doping for a sheet of size {sheet_size}: {elapsed_time:.2f} seconds")
    #
    # # Plot the stacked structure
    # stacked_graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Save the structure to a .xyz file
    # write_xyz(stacked_graphene.graph, "ABC_stacking.xyz")

    ####################################################################################################################
    # CREATE A CNT STRUCTURE

    # cnt = CNT(bond_length=1.42, tube_length=10.0, tube_size=8, conformation="zigzag", periodic=False)
    cnt = CNT(bond_length=1.42, tube_length=10.0, tube_diameter=6, conformation="zigzag", periodic=False)
    cnt.add_nitrogen_doping(total_percentage=10)
    cnt.plot_structure(with_labels=True, visualize_periodic_bonds=False)

    # Save the CNT structure to a file
    write_xyz(cnt.graph, "CNT_structure_zigzag.xyz")

    ####################################################################################################################
    # CREATE A PORE STRUCTURE
    # Define parameters for the graphene sheets and CNT
    bond_length = 1.42  # Bond length for carbon atoms
    sheet_size = (20, 20)  # Size of the graphene sheets
    tube_length = 10.0  # Length of the CNT
    # tube_size = 8  # Number of hexagonal units around the CNT circumference
    tube_diameter = 7  # Diameter of the CNT
    conformation = "zigzag"  # Conformation of the CNT (can be "zigzag" or "armchair")

    # Create a Pore structure
    pore = Pore(
        bond_length=bond_length,
        sheet_size=sheet_size,
        tube_length=tube_length,
        # tube_size=tube_size,
        tube_diameter=tube_diameter,
        conformation=conformation,
    )

    # Add optional nitrogen doping (if needed)
    # pore.add_nitrogen_doping(total_percentage=10)

    # Visualize the structure with labels (without showing periodic bonds)
    pore.plot_structure(with_labels=True, visualize_periodic_bonds=False)

    # Save the Pore structure to a file
    write_xyz(pore.graph, "pore_structure.xyz")


if __name__ == "__main__":
    main()
