# unet-v2: how-to inference + generic weights for CPU + ONNX weights


Run ./serve/serve.py

1. download demo image and demo model weights (model was trained in Supervisely on lemons dataset)
2. transform original model weights (Data parallel) to generic weights 
   
`torch.nn.DataParallel` is a model wrapper that enables parallel GPU utilization. Code converts and saves a DataParallel model generically => you have the flexibility to load the model any way you want to any device you want.
   
3. Convert, save, validate weights to ONNX format
4. Compare that torch and onnx predictions are equal
