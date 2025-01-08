import triton_python_backend_utils as pb_utils
import torch
import torch.nn.functional as F
from typing import Dict
from transformers import pipeline
import numpy as np
import time
import os
import json

class TritonPythonModel:
    
    def initialize(self, args:Dict[str, str]) -> None:
        self.logger = pb_utils.Logger
        path: str = os.path.join(args['model_repository'], args['model_version'])
        # path: str = 'dbmdz/bert-large-cased-finetuned-conll03-english'

        self.unmasker = pipeline('token-classification', model=path, device=0)
    
                
    def execute(self, requests):
        responses = []
        start_time_batch = time.perf_counter()
        for request in requests:
            start_time = time.perf_counter()
            
            # Assuming the input tensor contains text for summarization
            text_input = pb_utils.get_input_tensor_by_name(request, "INPUT")
            text = text_input.as_numpy()[0].decode("utf-8")  # Assuming the text is encoded as bytes

            output = self.unmasker(text)
            optimized_output = [
                {
                    'entity': item['entity'],
                    'score': float(item['score']),  # You can specify float64 here, if needed
                    'index': item['index'],
                    'word': item['word'],
                    'start': item['start'],
                    'end': item['end']
                }
                for item in output
            ]

            result_json = json.dumps(optimized_output)

            # Prepare the inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT", np.array([result_json.encode()]))
                ]
            )

            self.logger.log_info(f"Time taken for single request: {time.perf_counter() - start_time}")
            responses.append(inference_response)

        self.logger.log_info(f"Time taken by batch: {time.perf_counter() - start_time_batch}")
        return responses
    

    def finalize(self):
        pass  

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)