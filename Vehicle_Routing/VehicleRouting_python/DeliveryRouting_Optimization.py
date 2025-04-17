import time
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.spatial.distance import pdist, squareform

# -------------------- 1. SIMPLER DATA --------------------
np.random.seed(42)

num_nodes = 8                  # 0 = depot, 1-5 = customers
demands = np.random.randint(1, 7, size=num_nodes)
demands[0] = 0
truck_capacity = 10
num_vehicles = 4

coords = np.random.rand(num_nodes, 2)
dist_mat = np.round(squareform(pdist(coords)), 4)

# -------------------- 2. MODEL SETUP --------------------
model = pyo.ConcreteModel("SimpleCVRP")

model.Nodes = pyo.Set(initialize=range(num_nodes))
model.VehicleSet = pyo.Set(initialize=range(num_vehicles))
model.Arcs = pyo.Set(initialize=[(i, j) for i in model.Nodes for j in model.Nodes if i != j])

model.Cap = pyo.Param(initialize=truck_capacity)
model.dist = pyo.Param(model.Arcs, initialize={(i, j): dist_mat[i, j] for (i, j) in model.Arcs})
model.demand = pyo.Param(model.Nodes, initialize={i: d for i, d in enumerate(demands)})

model.useArc = pyo.Var(model.Arcs, model.VehicleSet, within=pyo.Binary)
model.visitNode = pyo.Var(model.Nodes, model.VehicleSet, within=pyo.Binary)

# -------------------- 3. CONSTRAINTS --------------------
def inbound_flow(m, i):
    if i == 0:
        return sum(m.useArc[n, i, v] for n in m.Nodes if n != i for v in m.VehicleSet) == len(m.VehicleSet)
    else:
        return sum(m.useArc[n, i, v] for n in m.Nodes if n != i for v in m.VehicleSet) == 1

def outbound_flow(m, i):
    if i == 0:
        return sum(m.useArc[i, n, v] for n in m.Nodes if n != i for v in m.VehicleSet) == len(m.VehicleSet)
    else:
        return sum(m.useArc[i, n, v] for n in m.Nodes if n != i for v in m.VehicleSet) == 1

def vehicle_in_assign(m, i, v):
    return sum(m.useArc[n, i, v] for n in m.Nodes if n != i) == m.visitNode[i, v]

def vehicle_out_assign(m, i, v):
    return sum(m.useArc[i, n, v] for n in m.Nodes if n != i) == m.visitNode[i, v]

def capacity_check(m, v):
    return sum(m.visitNode[i, v] * m.demand[i] for i in m.Nodes) <= m.Cap

model.Inbound = pyo.Constraint(model.Nodes, rule=inbound_flow)
model.Outbound = pyo.Constraint(model.Nodes, rule=outbound_flow)
model.VehicleIn = pyo.Constraint(model.Nodes, model.VehicleSet, rule=vehicle_in_assign)
model.VehicleOut = pyo.Constraint(model.Nodes, model.VehicleSet, rule=vehicle_out_assign)
model.CapCheck = pyo.Constraint(model.VehicleSet, rule=capacity_check)
model.SubtourElims = pyo.ConstraintList()

# -------------------- 4. OBJECTIVE --------------------
model.obj = pyo.Objective(
    expr=sum(model.useArc[i, j, v] * model.dist[i, j] for (i, j) in model.Arcs for v in model.VehicleSet),
    sense=pyo.minimize
)

# -------------------- 5. SOLVER --------------------
solver = SolverFactory("gurobi")
solver.options["MIPGap"] = 1e-6
solver.options["Heuristics"] = 0.2
solver.options["Symmetry"] = 1

# -------------------- 6. SUBTOUR HANDLING --------------------
def fetch_active_arcs(m):
    arcs_used = []
    for (i, j) in m.Arcs:
        for v in m.VehicleSet:
            if np.isclose(m.useArc[i, j, v].value, 1, atol=1e-1):
                arcs_used.append((i, j))
    return arcs_used

def fetch_current_tours(m):
    tours = []
    for v in m.VehicleSet:
        route = [0]
        node = 0
        while True:
            next_node = None
            for j in m.Nodes:
                if j != node and np.isclose(m.useArc[node, j, v].value, 1, atol=1e-1):
                    next_node = j
                    break
            if next_node is None or next_node == 0:
                break
            route.append(next_node)
            node = next_node
        route.append(0)
        if len(route) > 2:
            tours.append(route)
    return tours

def find_subtours(arcs):
    G = nx.DiGraph()
    G.add_edges_from(arcs)
    return list(nx.strongly_connected_components(G))

def add_subtour_constraints(m, sccs):
    changed = False
    all_nodes = set(m.Nodes)
    for s in sccs:
        if 0 not in s and len(s) > 1:
            changed = True
            outside = all_nodes - s
            for i in s:
                for v in m.VehicleSet:
                    m.SubtourElims.add(
                        m.visitNode[i, v] <= sum(m.useArc[j, k, v] for j in s for k in outside)
                    )
    return changed

def _subtour_step(m, verbose=True):
    solver.solve(m)
    arcs = fetch_active_arcs(m)
    sccs = find_subtours(arcs)
    if verbose:
        print("Current SCCs:", sccs)
    time.sleep(0.1)
    return add_subtour_constraints(m, sccs)

def iterative_solve(m, verbose=True):
    while True:
        updated = _subtour_step(m, verbose)
        if not updated:
            break

# -------------------- 7. SOLVE --------------------
iterative_solve(model, verbose=True)
tours = fetch_current_tours(model)

# -------------------- 8. PRINT --------------------
print("\nFinal Routes:")
for i, t in enumerate(tours):
    print(f" Vehicle {i}: {t}")

# -------------------- 9. VISUALIZE --------------------
cmap = plt.get_cmap("tab10")  # Use a more vibrant colormap
colors = cycle(cmap.colors)

fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
for idx, route in enumerate(tours):
    c = next(colors)
    delivery_nodes = np.array(route)
    ax.plot(coords[delivery_nodes, 0], coords[delivery_nodes, 1], color=c, label=f"Delivery Van {idx}", linewidth=2)
    ax.scatter(coords[delivery_nodes, 0], coords[delivery_nodes, 1], color=c, edgecolor='black', s=50)

ax.scatter(coords[0, 0], coords[0, 1], s=200, marker='s', edgecolor='black', label='Amazon Depot')
ax.set_title("Amazon Package Delivery Route Optimization")
ax.legend()
plt.tight_layout()
plt.show()
