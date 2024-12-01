from datetime import datetime
import grpc
from concurrent import futures
import time
import logging
import sys
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/app/generated')

from generated import embedding_service_pb2
from generated import embedding_service_pb2_grpc
from spatio_temporal_transformer.spatio_temporal_transformer_get_emb import SpatioTemporalTransformer, prepare_data
from spatio_temporal_transformer.node_embedding_processor import RealtimeNodeEmbeddingProcessor
from torch.utils.data import DataLoader
import torch


brokers_db = {}

def tensor_to_proto(tensor: np.ndarray) -> embedding_service_pb2.Tensor:
    """
    NumPy Tensor transform to Protobuf Tensor
    :param tensor: NumPy ndarray
    :return: Protobuf Tensor 
    """
    return embedding_service_pb2.Tensor(
        values=tensor.flatten().tolist(), 
        shape=list(tensor.shape)          
    )

def proto_to_tensor(proto: embedding_service_pb2.Tensor) -> np.ndarray:
    """
    Protobuf Tensor message transform to NumPy Tensor
    :param proto: Protobuf Tensor
    :return: NumPy ndarray
    """
    values = np.array(proto.values, dtype=np.float32) 
    shape = tuple(proto.shape)                        
    return values.reshape(shape)                      # reverse to NumPy ndarray


def call_model(broker_address: str) -> embedding_service_pb2.Tensor:
    """
    Call model
    :param broker_address: broker_address
    :return: Tensor
    """

    experiment_config = {
        'file_path': './preprocessed_data_sttransformer/176/merged_df.csv',
        'input_timesteps': 10,
        'forecast_horizon': 2,
        'target_node': "B2",
        'target_feature': "ResponseTime",
        'selected_features': ["Throughput", "ResponseTime"],
        'batch_size': 32,
        'num_epochs': 100,
        'patience': 10,
        'learning_rate': 1e-3,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dropout': 0.1
    }

    # Prepare data
    train_dataset, val_dataset, test_dataset, scaler_X, scaler_y, node_names, feature_names = prepare_data(
        experiment_config['file_path'],
        experiment_config['input_timesteps'],
        experiment_config['forecast_horizon'],
        experiment_config['target_node'],
        experiment_config['target_feature'],
        experiment_config['selected_features'],
        batch_size=experiment_config['batch_size']
    )

    train_loader = DataLoader(train_dataset, batch_size=experiment_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=experiment_config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=experiment_config['batch_size'])
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpatioTemporalTransformer(
        num_nodes=train_dataset.num_nodes,
        num_features=train_dataset.num_features,
        input_timesteps=experiment_config['input_timesteps'],
        forecast_horizon=experiment_config['forecast_horizon'],
        d_model=experiment_config['d_model'],
        nhead=experiment_config['nhead'],
        num_layers=experiment_config['num_layers'],
        dropout=experiment_config['dropout']
    ).to(device)

    sample_data, _ = next(iter(test_loader))
    sample_data = sample_data.to(device)
    
    embedding_result = model.get_node_embedding(sample_data)
    proto_tensor_result = tensor_to_proto(embedding_result)
    return proto_tensor_result

class EmbeddingServiceServicer(embedding_service_pb2_grpc.EmbeddingServiceServicer):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ""  # 학습된 
        self.realtime_node_embedding_processor = RealtimeNodeEmbeddingProcessor(
            model=model,
            device=device,
            num_brokers=5,                   # 브로커 수
            input_timesteps=10,              # 시계열 윈도우 크기(초)
            selected_features=[              # 사용할 feature 리스트
                "Cpu", "Memory", "Throughput", "ResponseTime"
            ]
        )


    def SendProssessLog(self, request: embedding_service_pb2.SendProssessLogRequest, context):
        logging.info(f"Received SendProssessLog request with broker_address: {request.broker_address}")
        def tick_proto_request_to_dict(tick_proto_request: embedding_service_pb2.TickLog) -> dict:

            return {
                "Cpu": tick_proto_request.Cpu,
                "Memory": tick_proto_request.Memory,
                "Throughput": tick_proto_request.Throughput,
                "ResponseTime": tick_proto_request.ResponseTime,
                "InterArrivalTime": tick_proto_request.InterArrivalTime,
                "QueueLength": tick_proto_request.QueueLength,
                "QueueTime": tick_proto_request.QueueTime,
                "ServiceTime": tick_proto_request.ServiceTime
            }
        
        def performance_info_to_dict(performance_info: embedding_service_pb2.PerformanceInfo) -> dict:
            return {
                "BrokerId": performance_info.BrokerId,
                "Cpu": performance_info.Cpu,
                "Memory": performance_info.Memory,
                "QueueLength": performance_info.QueueLength,
                "QueueTime": performance_info.QueueTime,
                "ResponseTime": performance_info.ResponseTime,
                "ServiceTime": performance_info.ServiceTime,
                "Throughput": performance_info.Throughput,
                "Timestamp": performance_info.Timestamp
            }
        
        def hop_proto_request_to_dict(hop_proto_request: embedding_service_pb2.HopLog) -> dict:
            performance_info_list: embedding_service_pb2.PerformanceInfo = hop_proto_request.PerformanceInfo
            performance_info_list = [performance_info_to_dict(performance_info) for performance_info in performance_info_list]
            return {
                "Node": hop_proto_request.Node,
                "PerformanceInfo": performance_info_list,
                "Msg": hop_proto_request.Msg,
                "Level": hop_proto_request.Level,
            }
        if request.type == "HopLog":
            self.realtime_node_embedding_processor.add_message(
                timestamp=datetime.now(), 
                hop_log=hop_proto_request_to_dict(request.HopLog)
            )
        elif request.type == "TickLog":
            self.realtime_node_embedding_processor.add_message(
                timestamp=datetime.now(), 
                tick_log=tick_proto_request_to_dict(request.TickLog)
            )
        else:
            logging.warning(f"Unknown request type: {request.type}")
            return embedding_service_pb2.SendProssessLogResponse()
        return embedding_service_pb2.SendProssessLogResponse()
    
    def CallModel(self, request: embedding_service_pb2.CallModelRequest, context):
        logging.info(f"Received CallModel request with broker_address: {request.broker_address}")
        brokers_db[request.broker_address] = {
            "broker_address": request.broker_address
        }

        embedding_result = call_model(request.broker_address)
        return embedding_service_pb2.CallModelResponse(
            tensor=embedding_result
        )


def serve():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embedding_service_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingServiceServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()
    

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为 INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    serve()

