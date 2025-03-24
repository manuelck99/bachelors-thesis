import random
from collections import defaultdict

import numpy as np
import pytest

import bachelors_thesis.util as util
import bachelors_thesis.vehicle_record as vr

random.seed(0)

DIMENSION = 256
NUMBER_OF_RECORDS = 10_000


@pytest.fixture(scope="class")
def vehicle_records():
    records = list()
    for i in range(NUMBER_OF_RECORDS):
        record = dict()

        record[vr.RECORD_ID] = i
        record[vr.VEHICLE_ID] = i
        record[vr.CAMERA_ID] = i
        record[vr.VEHICLE_FEATURE] = np.random.rand(DIMENSION).astype(np.float32)
        if random.random() < 0.5:
            record[vr.LICENSE_PLATE_FEATURE] = np.random.rand(DIMENSION).astype(np.float32)
        else:
            record[vr.LICENSE_PLATE_FEATURE] = None
        record[vr.LICENSE_PLATE_TEXT] = "AAAAAA" if random.random() < 0.3 else "BBBBBB"
        record[vr.TIMESTAMP] = i

        records.append(vr.VehicleRecord(record))

    return records


class TestVehicleRecordCluster:
    def test_centroid_vehicle_feature(self, vehicle_records):
        centroid_vehicle_feature = np.zeros(DIMENSION, dtype=np.float32)
        for record in vehicle_records:
            centroid_vehicle_feature += util.normalize(record.vehicle_feature)
        centroid_vehicle_feature /= len(vehicle_records)

        cluster = vr.VehicleRecordCluster(dimension=DIMENSION)
        for record in vehicle_records:
            cluster.add_record(record)

        assert np.allclose(centroid_vehicle_feature, cluster.centroid_vehicle_feature, atol=1e-6, rtol=0.0)
        assert np.allclose(cluster.centroid_vehicle_feature, centroid_vehicle_feature, atol=1e-6, rtol=0.0)

    def test_centroid_license_plate_feature(self, vehicle_records):
        centroid_license_plate_feature = np.zeros(DIMENSION, dtype=np.float32)
        number_of_license_plate_features = 0
        for record in vehicle_records:
            if record.has_license_plate():
                centroid_license_plate_feature += util.normalize(record.license_plate_feature)
                number_of_license_plate_features += 1
        centroid_license_plate_feature /= number_of_license_plate_features

        cluster = vr.VehicleRecordCluster(dimension=DIMENSION)
        for record in vehicle_records:
            cluster.add_record(record)

        assert np.allclose(centroid_license_plate_feature, cluster.centroid_license_plate_feature, atol=1e-6, rtol=0.0)
        assert np.allclose(cluster.centroid_license_plate_feature, centroid_license_plate_feature, atol=1e-6, rtol=0.0)

    def test_centroid_license_plate_text(self, vehicle_records):
        license_plate_texts = defaultdict(int)
        for record in vehicle_records:
            if record.has_license_plate():
                license_plate_texts[record.license_plate_text] += 1
        k = list(license_plate_texts.keys())
        v = list(license_plate_texts.values())
        centroid_license_plate_text = k[v.index(max(v))]

        cluster = vr.VehicleRecordCluster(dimension=DIMENSION)
        for record in vehicle_records:
            cluster.add_record(record)

        assert centroid_license_plate_text == cluster.calculate_centroid_license_plate_text()
