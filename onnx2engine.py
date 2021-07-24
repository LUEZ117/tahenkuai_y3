import tensorrt as trt
import torch

def build_engine(onnx_file_path, category_num=5, verbose=False):
    """Build a TensorRT engine from an ONNX file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
        builder.max_batch_size = 1 # TensorRT可以优化的最大的batch size，实际运行时，选择的batch size小于等于该值。
        builder.fp16_mode = True
        #builder.strict_type_constraints = True

        # Parse model file
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        if trt.__version__[0] >= '7':
            # The actual yolo*.onnx is generated with batch size 64.
            # Reshape input to batch size 1
            shape = list(network.get_input(0).shape)
            shape[0] = 1
            network.get_input(0).shape = shape

        print('Adding yolo_layer plugins...')
        model_name = onnx_file_path[:-5]
        network = add_yolo_plugins( # Add yolo plugins into a TensorRT network。
            network, model_name, category_num, TRT_LOGGER)

        print('Building an engine.  This would take a while...')
        print('(Use "--verbose" to enable verbose logging.)')
        engine = builder.build_cuda_engine(network) # Builds an ICudaEngine from a INetworkDefinition。
        print('Completed creating engine.')
        return engine

# Load pretrained model weights
model_url = 'best_y3.pt'
batch_size = 1    # just a random number
 
# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch.load_state_dict(model_url)
 
# set the model to inference mode
torch.eval()

# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)
 
# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})