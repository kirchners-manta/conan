from typing import Optional

import networkx as nx

from conan.playground.doping import DopingStructureCollection, NitrogenSpecies


class AtomLabeler:
    def __init__(self, graph: nx.Graph, doping_structures: Optional["DopingStructureCollection"] = None):
        # Validate the graph input
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"Expected graph to be a networkx Graph instance, but got {type(graph).__name__}.")

        self.graph = graph
        """The networkx graph representing the structure of the material (e.g., graphene sheet)."""

        # Validate the doping_structures input
        if doping_structures is not None and not isinstance(doping_structures, DopingStructureCollection):
            raise TypeError(
                f"Expected doping_structures to be a DopingStructureCollection instance or None, "
                f"but got {type(doping_structures).__name__}."
            )

        self.doping_structures = doping_structures
        """The collection of doping structures within the structure."""

    def label_atoms(self):
        """
        Label the atoms in the graphene structure based on their doping species and local environment.

        This method assigns labels to atoms based on the doping structures they belong to and their immediate
        environment.
        Atoms that are part of a doping structure get labeled according to their specific nitrogen or carbon species.
        In each doping cycle, the neighboring atoms of a C atom that are also within a cycle are also specified (_CC or
        _CN), as well as a graphitic-N neighbor outside the cycle, if present (_G).
        All other carbon atoms are labeled as "CG" for standard graphene carbon.

        In other words:
        Atoms in the same symmetrically equivalent environment get the same label, while those in different
        environments are labeled differently.
        """
        if not self.doping_structures:
            # Label all atoms as "CG" if there are no doping structures
            for node in self.graph.nodes:
                self.graph.nodes[node]["label"] = "CG"
            return

        # Loop through each doping structure and label the atoms
        for structure in self.doping_structures.structures:
            species = structure.species  # Get the nitrogen species (e.g., PYRIDINIC_1, PYRIDINIC_2, etc.)

            # Determine the appropriate labels for nitrogen and carbon atoms within the doping structure
            if species == NitrogenSpecies.GRAPHITIC:
                nitrogen_label = "NG"
                # Label nitrogen atom in GRAPHITIC species
                for atom in structure.nitrogen_atoms:
                    self.graph.nodes[atom]["label"] = nitrogen_label
            else:
                # For pyridinic species, use NP1, NP2, NP3, NP4 for nitrogen, and CP1, CP2, CP3, CP4 for carbon
                nitrogen_label = f"NP{species.value[-1]}"
                carbon_label_base = f"CP{species.value[-1]}"

                # Label nitrogen atoms within the doping structure
                for atom in structure.nitrogen_atoms:
                    self.graph.nodes[atom]["label"] = nitrogen_label

                # Label carbon atoms in the cycle of the doping structure
                for atom in structure.cycle:
                    if atom not in structure.nitrogen_atoms:
                        # Efficient one-liner to assign the label based on neighbors
                        cycle_neighbors = structure.subgraph.neighbors(atom)
                        self.graph.nodes[atom]["label"] = (
                            f"{carbon_label_base}_CC"
                            if all(self.graph.nodes[n]["element"] == "C" for n in cycle_neighbors)
                            else f"{carbon_label_base}_CN"
                        )

                        # Check for additional cases where a neighboring atom is Graphitic-N
                        neighbors = self.graph.neighbors(atom)
                        if any(
                            self.graph.nodes[n].get("nitrogen_species") == NitrogenSpecies.GRAPHITIC for n in neighbors
                        ):
                            self.graph.nodes[atom]["label"] += "_G"

        # Label remaining carbon atoms as "CG"
        for node in self.graph.nodes:
            if "label" not in self.graph.nodes[node]:  # If the node hasn't been labeled yet
                self.graph.nodes[node]["label"] = "CG"
