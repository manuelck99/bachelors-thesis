import logging
import time

import bachelors_thesis.clustering as clustering
import bachelors_thesis.config as config
from bachelors_thesis.vehicle_record import load_records

logger = logging.getLogger(__name__)

data_path = "datasets/UrbanVehicle/records/vehicles"


def run():
    records = load_records(
        [f"{data_path}/records-vehicle-{i}.json" for i in range(config.NUMBER_OF_LABELLED_VEHICLES)])
    logger.debug(f"Number of records: {len(records)}")

    t0 = time.time_ns()
    clusters = clustering.cluster_records(records)
    t1 = time.time_ns()

    print("Execution time [ms]: ", (t1 - t0) / 1000 / 1000)
    print("Number of clusters: ", len(clusters))


if __name__ == "__main__":
    run()
