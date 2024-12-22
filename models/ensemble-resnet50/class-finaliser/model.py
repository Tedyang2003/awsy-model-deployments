import os
import json
from typing import Dict, List
import numpy as np
import triton_python_backend_utils as pb_utils
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the model.
        :param args: arguments from Triton config file
        """
        # Load class names from JSON file

        path: str = os.path.join(args["model_repository"], args["model_version"])

        class_file_path = "imagenet-labels.json"  # Expecting a path to the JSON file
        with open(f"{path}/{class_file_path}", 'r') as f:
            self.class_names = json.load(f)

    def execute(self, requests) -> "List[pb_utils.InferenceResponse]":
        """
        Process the requests and produce outputs.
        :param requests: 1 or more requests received by Triton server.
        :return: list of inference responses
        """
        responses = []

        # Process each request
        for request in requests:
            # Initialize a list to hold top classifications from each classifier
            all_top_classes = []

            # Process the classifiers
            for i in range(1, 4):
                input_tensor = pb_utils.get_input_tensor_by_name(request, f"output_classifier_{i}")
                input_data = input_tensor.as_numpy() 
                
                # Get top classifications for this classifier
                top_indices = np.argsort(input_data[0])[::-1]  # Sort indices based on probabilities
                top_classes = self.class_names[top_indices[0]] # Top class name

                # Log the received output data and top classes
                logging.info(f"Output from classifier {i}: Shape: {input_data.shape}, Top classes: {top_classes}")

                # Append the top classes for this classifier
                all_top_classes.append(top_classes)

            # Prepare output: You can choose how to structure this output
            logging.info(f"All top classes {all_top_classes}")

            output_tensor = pb_utils.Tensor("final_output", np.array(all_top_classes).astype(object))

            # Create inference response
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

