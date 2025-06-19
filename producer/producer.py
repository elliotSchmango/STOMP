import os
import json
import time
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

DATA_PATH = "/data/test_FD001.txt"

def connect_to_kafka():
    while True:
        try:
            print("Trying to connect to Kafka...")
            producer = KafkaProducer(
                bootstrap_servers='kafka:9092',
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Connected to Kafka")
            return producer
        except NoBrokersAvailable:
            print("Kafka connection error... retrying in 2 seconds.")
            time.sleep(2)

#replaced with real data file
def parse_line(line):
    parts = line.strip().split()
    engine_id = int(parts[0])
    cycle = int(parts[1])
    sensors = [float(val) for val in parts[5:]]  # skip op settings
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        "sensors": sensors
    }

#create kafka connection
producer = connect_to_kafka()

#stream data every half second
with open(DATA_PATH, "r") as f:
    for line in f:
        record = parse_line(line)
        producer.send('engine.telemetry', value=record)
        print(f"Sent: {record}")
        time.sleep(0.05)