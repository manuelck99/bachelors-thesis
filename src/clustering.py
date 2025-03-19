import logging

import numpy as np
from faiss import normalize_L2, IndexFlatIP, IndexIDMap, omp_set_num_threads, omp_get_max_threads

logger = logging.getLogger(__name__)


def search(features,
           features_record_ids,
           k,
           dimension,
           number_of_threads):
    logger.debug("Creating vector database")
    db_features = np.vstack(features)

    logger.debug("Normalizing vector database")
    normalize_L2(db_features)

    logger.debug("Building FAISS index")
    features_index = IndexIDMap(IndexFlatIP(dimension))

    logger.debug("Adding vector database to FAISS index")
    features_index.add_with_ids(db_features, features_record_ids)

    logger.debug("Setting number of threads")
    if number_of_threads > omp_get_max_threads():
        omp_set_num_threads(omp_get_max_threads())
    else:
        omp_set_num_threads(number_of_threads)

    logger.debug("Similarity searching vector database")
    _, features_results = features_index.search(db_features, k)

    return features_results
