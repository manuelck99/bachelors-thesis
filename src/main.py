import logging
import time

from clustering import search
from util import load_records

K = 128
DIMENSIONS = 256
NUMBER_OF_THREADS = 8
SIMILARITY_THRESHOLD = 0.8

logger = logging.getLogger(__name__)

data_path = "../datasets/UrbanVehicle/records/cameras/records-camera-0.json"


def run():
    """
    Load records, assign each a numerical id, turn them into VehicleRecords and also keep track of the ids,
    making sure that you have two lists: one with VehicleRecords and other with ids all in corresponding order.
    Don't forget you have to separate VehicleRecords with license plates and those without.

    Feed them into the clustering search and it will return for each id, which VehicleRecords are most similar.
    Filter the record itself out.

    Merge the most similar VehicleRecords for each VehicleRecord

    Find the clusters of the candidate VehicleRecords and calculate their centroids

    Calculate the visual similarity between the VehicleRecord and the centroids of its candidate clusters

    Take the cluster with highest visual similarity and add the record to it (recalculate centroid)

    Do this for all VehicleRecords
    """

    records = load_records([data_path])
    print("Number of records:", len(records))
    records_dict = {record.record_id: record for record in records}

    vehicle_features = [record.vehicle_feature for record in records]
    vehicle_features_record_ids = [record.record_id for record in records]
    license_plate_features = [record.license_plate_feature for record in records if
                              record.license_plate_feature is not None]
    license_plate_features_record_ids = [record.record_id for record in records if
                                         record.license_plate_feature is not None]

    t0 = time.time_ns()
    vehicle_features_results, license_plate_features_results = search(vehicle_features, vehicle_features_record_ids,
                                                                      license_plate_features,
                                                                      license_plate_features_record_ids, K, DIMENSIONS,
                                                                      NUMBER_OF_THREADS)
    t1 = time.time_ns()
    print("Similarity search time [ms]:", (t1 - t0) / 1000 / 1000)

    candidate_records_dict = dict()
    for most_similar_records_ids, record_id in zip(vehicle_features_results, vehicle_features_record_ids):
        candidate_records_dict[record_id] = {records_dict[i] for i in most_similar_records_ids if i != record_id}

    for most_similar_records_ids, record_id in zip(license_plate_features_results, license_plate_features_record_ids):
        candidate_records_dict[record_id].union({records_dict[i] for i in most_similar_records_ids if i != record_id})


if __name__ == "__main__":
    run()
