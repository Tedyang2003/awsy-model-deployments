import triton_python_backend_utils as pb_utils
import torch
import torch.nn.functional as F
from typing import Dict
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time
import os

class TritonPythonModel:
    
    def initialize(self, args:Dict[str, str]) -> None:
        self.logger = pb_utils.Logger
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float324
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path: str = os.path.join(args['model_repository'], args['model_version'])
        # path: str = 'sentence-transformers/all-distilroberta-v1'

        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype = torch_dtype
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


            input = self.tokenizer([text], max_length=512, truncation=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**input)

            sentence_embeddings = self.mean_pooling(model_output, input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            self.logger.log_info(f"Embeddings: {sentence_embeddings}")
            
            sentence_embeddings = sentence_embeddings.cpu().numpy()
            # Prepare the inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT", np.array(sentence_embeddings))
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