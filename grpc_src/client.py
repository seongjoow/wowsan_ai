import grpc
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/app/generated')

from generated import embedding_service_pb2_grpc
from generated import embedding_service_pb2


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
                        Timestamp='2024-11-10 18:10:22.0186605'
                    ),
                ],
                Level='info',
                Msg='Advertisement(apple > 0.45) message-id-123'
            ),
            # TickLog={}
        )

        send_hop_logs_response = stub.SendProssessLog(send_hop_logs_request)
        print(f"Sent hop log: {send_hop_logs_response}")

        send_tick_logs_request = embedding_service_pb2.SendProssessLogRequest(
            broker_address="broker_address2",
            type="TickLog",
            # HopLog=embedding_service_pb2.HopLog(),
            TickLog=embedding_service_pb2.TickLog(
                Cpu=58.342480,
                Memory=23302144.0,
                QueueLength=0,
                QueueTime=0.300874,
                ServiceTime=0.000000,
                ResponseTime=0.300874,
                InterArrivalTime=0.000000,
                Throughput=3.323645
            )
        )
        send_tick_logs_response = stub.SendProssessLog(send_tick_logs_request)
        print(f"Sent tick log: {send_tick_logs_response}")

        return 0

    except grpc.RpcError as e:
        print(f"gRPC Error: {e.details()}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run())
