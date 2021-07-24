import torchvision.models as models
import torch
import torch.onnx

# Load pretrained model weights
model_url = 'best_y3.pt'
batch_size = 1    # just a random number
 
# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
ckpt = torch.load(model_url, map_location=map_location)
model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()

# set the model to inference mode
model.eval()

# Input to the model
x = torch.randn(batch_size, 3, 640, 640)
torch_out = model(x)
 
# Export the model
torch.onnx.export(model, x, "best_y3.onnx", verbose=False)
# torch.onnx.export(model,               # model being run
#                   x,                         # model input (or a tuple for multiple inputs)
#                   "best_y3.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                 'output' : {0 : 'batch_size'}})
print("Export Complete.")