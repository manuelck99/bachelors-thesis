import zmq

import networking_pb2

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5555")

    for _ in range(1_000_000):
        envelope = networking_pb2.Envelope()

        envelope.cluster.cluster_id = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        envelope.cluster.centroid_license_plate_text = "BBBBBBBBBBBBBB"
        envelope.cluster.centroid_vehicle_feature.extend([i for i in range(500)])
        envelope.cluster.centroid_license_plate_feature.extend([i for i in range(500)])
        envelope.cluster.node_path.extend([i for i in range(600)])

        for _ in range(1000):
            record = envelope.cluster.records.add()
            record.record_id = "hello"
            record.camera_id = 5
            record.vehicle_id = -1
            record.timestamp = -666

        socket.send(envelope.SerializeToString())

    envelope = networking_pb2.Envelope()
    envelope.done = True
    socket.send(envelope.SerializeToString())
