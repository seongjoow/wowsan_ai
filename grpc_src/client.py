from datetime import datetime
import time
import grpc
import sys
import os
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/app/generated')

from generated import embedding_service_pb2_grpc
from generated import embedding_service_pb2


def send_hop_logs(stub):
    """持续发送 HopLog"""
    while True:
        send_hop_logs_request = embedding_service_pb2.SendProssessLogRequest(
            broker_address="broker_address1",
            type="HopLog",
            HopLog=embedding_service_pb2.HopLog(
                Node='localhost:50002',
                PerformanceInfo=[
                    embedding_service_pb2.PerformanceInfo(
                        BrokerId='localhost:50003',
                        Cpu=58.481087,
                        Memory=22659072,
                        QueueLength=0,
                        QueueTime=0.109688,
                        ServiceTime=0.0,
                        ResponseTime=0.109688,
                        InterArrivalTime=0.0,
                        Throughput=9.116759,
                        Timestamp=str(datetime.now()),
                        Time=str(datetime.now()),
                    ),
                ],
                Level='info',
                Msg='Advertisement(apple > 0.45) message-id-123'
            )
        )
        send_hop_logs_response = stub.SendProssessLog(send_hop_logs_request)
        print(f"Sent hop log: {send_hop_logs_response}")
        time.sleep(0.1)  # 每 0.1 秒发送一次

def send_tick_logs(stub: embedding_service_pb2_grpc.EmbeddingServiceStub):
    """Every 1 second send a TickLog"""
    while True:
        send_tick_logs_request = embedding_service_pb2.SendProssessLogRequest(
            broker_address="broker_address2",
            type="TickLog",
            TickLog=embedding_service_pb2.TickLog(
                Cpu=58.342480,
                Memory=23302144.0,
                QueueLength=0,
                QueueTime=0.300874,
                ServiceTime=0.0,
                ResponseTime=0.300874,
                InterArrivalTime=0.0,
                Throughput=3.323645,
                Time=str(datetime.now().replace(microsecond=0)),
            )
        )
        send_tick_logs_response = stub.SendProssessLog(send_tick_logs_request)
        print(f"Sent tick log: {send_tick_logs_response}")
        time.sleep(1)  # 每 1 秒发送一次



def run():
    try:
        channel = grpc.insecure_channel('host.docker.internal:50051')
        stub = embedding_service_pb2_grpc.EmbeddingServiceStub(channel)

        # Call Model
        print("Call model...")
        call_model_request = embedding_service_pb2.CallModelRequest(
            broker_address="localhost:50051"
        )
        create_model_response = stub.CallModel(call_model_request)
        print(f"Called model: {create_model_response}")

        # Send Logs
        print("Sending logs...")

        hop_log_thread = threading.Thread(target=send_hop_logs, args=(stub,))
        tick_log_thread = threading.Thread(target=send_tick_logs, args=(stub,))

        hop_log_thread.start()
        tick_log_thread.start()

        hop_log_thread.join()
        tick_log_thread.join()

        return 0

    except grpc.RpcError as e:
        print(f"gRPC Error: {e.details()}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run())
