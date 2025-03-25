import logging
import random
import time

import bachelors_thesis.clustering as clst
import bachelors_thesis.config as cfg
import bachelors_thesis.evaluation as eval
import bachelors_thesis.vehicle_record as vr

logger = logging.getLogger(__name__)

data_path = "datasets/UrbanVehicle/records/vehicles"


def run():
    records = vr.load_records(
        [f"{data_path}/records-vehicle-{i}.json" for i in range(cfg.NUMBER_OF_LABELLED_VEHICLES)])
    logger.debug(f"Number of records: {len(records)}")

    random.shuffle(records)

    t0 = time.time_ns()
    clusters = clst.cluster_records(records)
    t1 = time.time_ns()

    print("Clustering execution time [ms]: ", (t1 - t0) / 1000 / 1000)
    print("Number of clusters: ", len(clusters))

    print()

    precision, recall, f1_score, expansion = eval.yu_ao_yan_evaluation(clusters)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1_score)
    print("Expansion: ", expansion)


if __name__ == "__main__":
    run()
