import os
import sys

from apscheduler.events import JobExecutionEvent, EVENT_JOB_ERROR, EVENT_JOB_MISSED, \
    EVENT_JOB_MAX_INSTANCES
from apscheduler.schedulers.blocking import BlockingScheduler

from edge.producer import Producer


def keep_alive():
    global producer
    producer.keep_alive()


producer = None


def add_job():
    scheduler.add_job(keep_alive, 'interval', seconds=2)


def scheduler_listener(event):
    global producer
    print("scheduler exception")
    if isinstance(event, JobExecutionEvent):
        print(event.exception)
    scheduler.remove_all_jobs()
    if 'MACOS' in os.environ and os.environ['MACOS']:
        producer = Producer(server_host='localhost:2181', distributor_kafka_host='localhost:9092')
    else:
        producer = Producer()
    add_job()


if __name__ == '__main__':
    print(sys.argv)
    if 'MACOS' in os.environ and os.environ['MACOS']:
        producer = Producer(server_host='40.253.65.179:2181', distributor_kafka_host='40.253.65.179:9092')
    elif len(sys.argv)>2:
        producer = Producer(server_host=sys.argv[1], distributor_kafka_host=sys.argv[2])
    else:
        producer = Producer()
    scheduler = BlockingScheduler()
    scheduler.add_listener(scheduler_listener, EVENT_JOB_MAX_INSTANCES | EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    add_job()

    try:
        scheduler.start()
    except(KeyboardInterrupt, SystemExit) as e:
        scheduler.shutdown()
