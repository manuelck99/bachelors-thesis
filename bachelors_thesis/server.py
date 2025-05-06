import time

import zmq

import networking_pb2

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://localhost:5555")

    done = False
    l = list()

    t0 = time.time_ns()
    while not done:
        message = socket.recv()
        envelope = networking_pb2.Envelope()
        envelope.ParseFromString(message)

        if envelope.WhichOneof("content") == "done":
            done = True
        else:
            l.append(envelope.cluster)

    print((time.time_ns() - t0) / 1000 / 1000)
    print(len(l))
