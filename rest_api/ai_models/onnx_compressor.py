from fastT5 import generate_onnx_representation, quantize

custom_model_path = "t5/custom"
pretrained_model_path = "t5/pretrained"
output_custom_model_path = "t5/onnx/custom"
output_pretrained_model_path = "t5/onnx/pretrained"

onnx_custom_model_paths = generate_onnx_representation(f"{custom_model_path}/model",
                                                       output_path=output_custom_model_path)
onnx_pretrained_model_paths = generate_onnx_representation(f"{pretrained_model_path}/model",
                                                           output_path=output_pretrained_model_path)

quant_custom_model_paths = quantize(onnx_custom_model_paths)
quant_pretrained_model_paths = quantize(onnx_pretrained_model_paths)

