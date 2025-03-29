import logging
import os
import random
import time

import bachelors_thesis.clustering as clst
import bachelors_thesis.config as cfg
import bachelors_thesis.evaluation as eval
import bachelors_thesis.vehicle_record as vr

logger = logging.getLogger(__name__)

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./datasets/UrbanVehicle/records/vehicles")

random.seed(0)


def run():
    records = vr.load_records(
        [f"{dataset_path}/records-vehicle-{i}.json" for i in range(cfg.NUMBER_OF_LABELLED_VEHICLES)])
    logger.info(f"Number of records: {len(records)}")

    random.shuffle(records)

    t0 = time.time_ns()
    clusters = clst.cluster_records(records)
    t1 = time.time_ns()

    logger.info(f"Clustering execution time [ms]: {(t1 - t0) / 1000 / 1000}")
    logger.info(f"Number of clusters: {len(clusters)}")

    precision, recall, f1_score, expansion = eval.yu_ao_yan_evaluation(clusters)
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-Score: {f1_score}")
    logger.info(f"Expansion: {expansion}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run()
