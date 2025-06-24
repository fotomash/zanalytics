
import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

sqs = boto3.client(
    "sqs",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

QUEUE_URL = os.getenv("SQS_METRICS_URL")

def process_metric(metric):
    print(f"[ZDX-METRIC] Account {metric['account']} → Equity: {metric['equity']} | Margin: {metric['margin']}")

def listen():
    while True:
        msgs = sqs.receive_message(QueueUrl=QUEUE_URL, MaxNumberOfMessages=10, WaitTimeSeconds=20)
        for m in msgs.get("Messages", []):
            payload = json.loads(m["Body"])
            if payload["type"] == "AccountMetrics":
                for metric in payload["payload"]["metrics"]:
                    process_metric(metric)
            sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=m["ReceiptHandle"])

if __name__ == "__main__":
    print("[ZDX-METRIC] Listening for live metrics...")
    listen()
