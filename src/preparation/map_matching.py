import json
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher


data_path = "../../datasets/UrbanVehicle"


def load_graph():
    path = f"{data_path}/map.json"

    map = None
    with open(path, "r") as file:
        map = json.load(file)

    nodes = []
    edges = []
    for map_item in map:
        if map_item["type"] == "node":
            nodes.append(map_item)
        elif map_item["type"] == "way":
            edges.append(map_item)
        else:
            pass
    
    road_graph = InMemMap(
        name="road_graph",
        use_latlon=False,
        use_rtree=True,
        index_edges=True
    )
    
    for node in nodes:
        road_graph.add_node(node["id"], (node["xy"][1], node["xy"][0]))
    
    for edge in edges:
        if edge["oneway"]:
            prev_node = edge["nodes"][0]
            for curr_node in edge["nodes"][1:]:
                road_graph.add_edge(prev_node, curr_node)
                prev_node = curr_node
        else:
            prev_node = edge["nodes"][0]
            for curr_node in edge["nodes"][1:]:
                road_graph.add_edge(prev_node, curr_node)
                road_graph.add_edge(curr_node, prev_node)
                prev_node = curr_node
    
    return road_graph


road_graph = load_graph()

trajectories = []
with open(f"{data_path}/trajectories.json", "r") as file:
    for line in file:
        trajectory = json.loads(line)
        trajectory = trajectory["xyt"]
        f = lambda l: l[2]
        trajectory.sort(key=f)
        trajectory = [(coord[1], coord[0]) for coord in trajectory]
        trajectories.append(trajectory)

for index, trajectory in enumerate(trajectories):
    matcher = DistanceMatcher(road_graph, max_dist=100, max_dist_init=200, obs_noise=50, max_lattice_width=10)
    states, last_matched_index = matcher.match(trajectory)
