import json
import os
from powerformer_encoder import PowerformerEncoder, load_layer_weights

# Configuration
MODEL_DIR = "./student_powerformer_rte" 
CONFIG_FILE = "powerformer_encoder_config.json"

def handle_client_data(engine_instance, encrypted_input, public_key, eval_key):
    print(f"Received data. Starting FHE inference with model from {MODEL_DIR}")

    config_path = os.path.join(MODEL_DIR, CONFIG_FILE)
    if not os.path.exists(config_path):
        print(f"ERROR: Powerformer config file not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        powerformer_config = json.load(f)

    num_hidden_layers = powerformer_config.get("num_hidden_layers")
    if num_hidden_layers is None:
        print(f"ERROR: 'num_hidden_layers' not found in powerformer_encoder_config.json")
        return None

    encoder_layers = []
    params_for_encoder = {
        "brpmax": powerformer_config["brpmax"],
        "relu_poly": powerformer_config["relu_poly"]
    }

    for i in range(num_hidden_layers):
        layer_dir = os.path.join(MODEL_DIR, f"layer{i:02d}")
        if not os.path.isdir(layer_dir):
            print(f"ERROR: Layer weights directory not found: {layer_dir}")
            return None
        
        layer_weights = load_layer_weights(layer_dir, engine_instance)
        
        encoder = PowerformerEncoder(
            engine=engine_instance,
            evk=eval_key,
            params=params_for_encoder, 
            w=layer_weights
        )
        encoder_layers.append(encoder)

    current_ct = encrypted_input
    for idx, layer in enumerate(encoder_layers):
        current_ct = layer(current_ct)

    print(f"FHE Inference Complete.")
    return current_ct

if __name__ == '__main__':
    print(f"server.py is not meant to be run directly without a client providing FHE objects.")

    # Example of how you might mock (very simplified):
    # class MockFHEObject: pass
    # class MockEngine:
    #     def encode(self, arr, level, padding=True): return MockFHEObject()
    #     def decode(self, ct, level, is_real=True): return []
    # engine = MockEngine()
    # pk, evk = MockFHEObject(), MockFHEObject()
    # encrypted_input_mock = MockFHEObject()
    #
    # # Ensure MODEL_DIR and layer directories with dummy .npy files and config exist for this test
    # # For example, create student_powerformer_rte_2/powerformer_encoder_config.json
    # # and student_powerformer_rte_2/layer00/, student_powerformer_rte_2/layer01/
    # # with dummy WQ.npy etc.
    #
    # # result = handle_client_data(engine, encrypted_input_mock, pk, evk)
    # # if result:
    # #     print("Mock server run successful, result:", type(result))
    # # else:
    # #     print("Mock server run failed.")
