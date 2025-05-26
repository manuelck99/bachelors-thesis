from argparse import ArgumentParser

from evaluation import load_clusters, cluster_evaluation_with_cluster_gt, trajectory_evaluation_with_cluster_gt
from util import load_graph, load

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--clusters-path",
        type=str,
        required=True,
        help="Path to the clusters"
    )
    parser.add_argument(
        "--clusters-gt-path",
        type=str,
        required=True,
        help="Path to the clusters ground-truth"
    )
    parser.add_argument(
        "--road-graph-path",
        type=str,
        required=True,
        help="Path to the road graph file"
    )
    parser.add_argument(
        "--cameras-info-path",
        type=str,
        required=True,
        help="Path to the cameras information file"
    )
    args = parser.parse_args()

    clusters = load_clusters(args.clusters_path)
    clusters_gt = load_clusters(args.clusters_gt_path)
    road_graph = load_graph(args.road_graph_path)
    cameras_info = load(args.cameras_info_path)

    precision, recall, f1_score, expansion = cluster_evaluation_with_cluster_gt(clusters_gt, clusters)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1_score}")
    print(f"Expansion: {expansion}")

    # TODO: Try tuning gamma and epsilon
    lcss, edr, stlc = trajectory_evaluation_with_cluster_gt(clusters_gt,
                                                            clusters,
                                                            road_graph,
                                                            cameras_info,
                                                            gamma=0.8,
                                                            epsilon=200)
    print(f"LCSS distance: {lcss}")
    print(f"EDR distance: {edr}")
    print(f"STLC distance: {stlc}")
