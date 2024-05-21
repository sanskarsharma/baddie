import os
import boto3
from abc import ABC, abstractmethod


class BedrockLLM(ABC):
    def __init__(self):
        self.bedrock = boto3.client(
            service_name='bedrock',
            region_name=os.environ['AWS_REGION'], 
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )

        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ['AWS_REGION'], 
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )

    def __list_foundation_models(self):
        return self.bedrock.list_foundation_models()

    @abstractmethod
    def get_response(self, user_prompt: str, stream_response: bool) -> str:
        pass
