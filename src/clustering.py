import logging

import numpy as np
from faiss import normalize_L2, IndexFlatIP, IndexIDMap, omp_set_num_threads, omp_get_max_threads

logger = logging.getLogger(__name__)


def search(vehicle_features,
           vehicle_features_record_ids,
           license_plate_features,
           license_plate_features_record_ids,
           k,
           dimensions,
           number_of_threads):
    logger.debug("Creating vector databases")
    db_vehicle_features = np.vstack(vehicle_features)
    db_license_plate_features = np.vstack(license_plate_features)

    logger.debug("Normalizing vector databases")
    normalize_L2(db_vehicle_features)
    normalize_L2(db_license_plate_features)

    logger.debug("Building FAISS indices")
    vehicle_features_index = IndexIDMap(IndexFlatIP(dimensions))
    license_plate_features_index = IndexIDMap(IndexFlatIP(dimensions))

    logger.debug("Adding vector databases to FAISS indices")
    vehicle_features_index.add_with_ids(db_vehicle_features, vehicle_features_record_ids)
    license_plate_features_index.add_with_ids(db_license_plate_features, license_plate_features_record_ids)

    logger.debug("Setting number of threads")
    if number_of_threads > omp_get_max_threads():
        omp_set_num_threads(omp_get_max_threads())
    else:
        omp_set_num_threads(number_of_threads)

    logger.debug("Similarity searching vector databases")
    _, vehicle_features_results = vehicle_features_index.search(db_vehicle_features, k)
    _, license_plate_features_results = license_plate_features_index.search(db_license_plate_features, k)

    return vehicle_features_results, license_plate_features_results
