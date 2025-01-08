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
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float324
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path: str = os.path.join(args['model_repository'], args['model_version'])
        # path: str = 'sshleifer/distilbart-cnn-12-6'

        self.model = BartForConditionalGeneration.from_pretrained(
            path,
            torch_dtype = torch_dtype,
        )
        self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            path
        )
        
                
    def execute(self, requests):
        responses = []
        start_time_batch = time.perf_counter()
        for request in requests:
            start_time = time.perf_counter()
            
            # Assuming the input tensor contains text for summarization
            text_input = pb_utils.get_input_tensor_by_name(request, "INPUT")
            text = text_input.as_numpy()[0].decode("utf-8")  # Assuming the text is encoded as bytes


            inputs = self.tokenizer([text], max_length=1024, truncation=True, return_tensors="pt").to(self.device)
            

            # You can adjust max_length and min length as needed
            summary_ids = self.model.generate(
                inputs["input_ids"]
            )
            # Decode the generated summary
            summary_text = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            self.logger.log_info(f"Summary: {summary_text}...")
            
            # Prepare the inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT", np.array([summary_text.encode()]))  # Encode back to bytes
                ]
            )

            self.logger.log_info(f"Time taken for single request: {time.perf_counter() - start_time}")
            responses.append(inference_response)

        self.logger.log_info(f"Time taken by batch: {time.perf_counter() - start_time_batch}")
        return responses
    

    def finalize(self):
        pass  