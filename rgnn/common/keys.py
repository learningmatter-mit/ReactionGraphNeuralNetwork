"""Keys used in the input or output dictionary.
These are just strings, but they are defined as Final types to make it easier to
use them in type annotations and more readable.
For example, instead of writing:
    def forward(self, atoms_graph: AtomsGraphDict, output: OutputDict) -> OutputDict:
        output["force"] = self.compute_output(atoms_graph, output)
        return output

We can write:
    from neural_iap.common import keys as K
    def forward(self, atoms_graph: AtomsGraphDict, output: OutputDict) -> OutputDict:
        output[K.force] = self.compute_output(atoms_graph, output)
        return output

## Symbol definition
- `B`: The number of batches.
- `N`: The number of atoms.
- `M`: The number of total edges(=neighbors) in graph.

## Key descriptions
- pos: The positions of the atoms. Shape: `(N, 3)` Dtype: `float32`
- cell: The cell vectors. Shape: `(3, 3)` or `(B, 3, 3)` Dtype: `float32`
- elems: The atomic numbers of the atoms. Shape: `(N,)` Dtype: `int64`
- energy: The energy of the system. Shape: `(B,)` or `()` Dtype: `float32`
- force: The force on the atoms. Shape: `(N, 3)` or `(N, 3)` Dtype: `float32`
- n_atoms: The number of atoms in each batch. Shape: `(B,)` Dtype: `int64`
- batch: The batch index of each atom. ex) `[0, 0, 1, 1, 1]` Shape: `(N,)` Dtype: `int64`
- edge_index: The edge index of the graph. First row is the source node, and the second row is the target node.
    Note that, as message flows from source to target, the edge index is reversed from the usual definition
    when edge index represents neighbors: src is neighbor idx and dst is center idx.
    Shape: `(2, M)` Dtype: `int64`
- edge_shift: The 3D shift vectors on pos for each edge.
    This usually means the offset of neighbor atom position raised by periodic boundary condition.
    Shape: `(M, 3)` Dtype: `float32`
- edge_vec: The 3D vector connecting edges of the graph.
    This usually means the displacement vector from center atom to neighbor atom.
    Shape: `(M, 3)` Dtype: `float32`
- node_features: The node features of the graph. Shape: `(N, F)` Dtype: `float32`
- edge_features: The edge features of the graph. Shape: `(M, F)` Dtype: `float32`
- global_features: The global features of the graph. Shape: `(B, F)` or `(F,)` Dtype: `float32`
- node_vec_features: The node vector features of the graph. Shape: `(N, F, G)` Dtype: `float32`
- edge_vec_features: The edge vector features of the graph. Shape: `(M, F, G)` Dtype: `float32`
- global_vec_features: The global vector features of the graph. Shape: `(B, F, G)` or `(F, G)` Dtype: `float32`
"""
from typing import Final

pos: Final["pos"] = "pos"
cell: Final["cell"] = "cell"
elems: Final["elems"] = "elems"
atomic_energy: Final["atomic_energy"] = "atomic_energy"
energy: Final["energy"] = "energy"
edge_energy: Final["edge_energy"] = "edge_energy"
energy_pred: Final["energy_pred"] = "energy_pred"
force: Final["force"] = "force"
stress: Final["stress"] = "stress"
hessian: Final["hessian"] = "hessian"
n_atoms: Final["n_atoms"] = "n_atoms"

batch: Final["batch"] = "batch"
edge_index: Final["edge_index"] = "edge_index"
edge_shift: Final["edge_shift"] = "edge_shift"
edge_vec: Final["edge_vec"] = "edge_vec"
node_features: Final["node_features"] = "node_features"
edge_features: Final["edge_features"] = "edge_features"
global_features: Final["global_features"] = "global_features"
node_vec_features: Final["node_vec_features"] = "node_vec_features"
edge_vec_features: Final["edge_vec_features"] = "edge_vec_features"
global_vec_features: Final["global_vec_features"] = "global_vec_features"

# Reaction Keys
n_atoms_i: Final["n_atoms_i"] = "n_atoms_i"
n_atoms_f: Final["n_atoms_f"] = "n_atoms_f"
barrier: Final["barrier"] = "barrier"
freq: Final["freq"] = "freq"
energy_i: Final["energy_i"] = "energy_i"
energy_f: Final["energy_f"] = "energy_f"
delta_e: Final["delta_e"] = "delta_e"
reaction_features: Final["reaction_features"] = "reaction_features"
q0: Final["q0"] = "q0"
q1: Final["q1"] = "q1"
q2_i: Final["q2_i"] = "q2_i"
Q0: Final["Q0"] = "Q0"
Q1: Final["Q1"] = "Q1"
Q2_i: Final["Q2_i"] = "Q2_i"
rl_q: Final["rl_q"] = "rl_q"
dqn_feat: Final["dqn_feat"] = "dqn_feat"
