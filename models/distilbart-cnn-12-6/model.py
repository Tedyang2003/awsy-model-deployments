import triton_python_backend_utils as pb_utils
import torch
from typing import Dict
from transformers import BartForConditionalGeneration, AutoTokenizer, pipeline
import numpy as np
import time
import os

class TritonPythonModel:
    
    def initialize(self, args:Dict[str, str]) -> None:
        self.logger = pb_utils.Logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path: str = os.path.join(args['model_repository'], args['model_version'])
        # path: str = 'sshleifer/distilbart-cnn-12-6'
        self.summariser = pipeline('summarization', model=path, device=0)

                
    def execute(self, requests):
        responses = []
        start_time_batch = time.perf_counter()
        for request in requests:
            start_time = time.perf_counter()
            
            # Assuming the input tensor contains text for summarization
            text_input = pb_utils.get_input_tensor_by_name(request, "INPUT")
            text = text_input.as_numpy()[0].decode("utf-8")  # Assuming the text is encoded as bytes

            output = self.summariser(text)

            self.logger.log_info(f"Summary: {output}...")
            
            # Prepare the inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT", np.array([output[0]['summary_text'].encode()]))  # Encode back to bytes
                ]
            )

            self.logger.log_info(f"Time taken for single request: {time.perf_counter() - start_time}")
            responses.append(inference_response)

        self.logger.log_info(f"Time taken by batch: {time.perf_counter() - start_time_batch}")
        return responses
    

    def finalize(self):
        pass  