from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import json
import time

def connect_to_kafka():
    while True:
        try:
            print("Trying to connect to Kafka...")
            producer = KafkaProducer(
                bootstrap_servers='kafka:9092',
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Connected to Kafka!")
            return producer
        except NoBrokersAvailable:
            print("Kafka not available yet... retrying in 3 seconds.")
            time.sleep(3)

producer = connect_to_kafka()
#now send telemetry data in intervals after connection is created
while True:
    telemetry = {
        "engine_id": 1,
        "cycle": int(time.time()) % 1000,
        "sensor_1": 642.3,
        "sensor_2": 129.6
    }
    producer.send('engine.telemetry', value=telemetry)
    print(f"Sent: {telemetry}")
    time.sleep(1)