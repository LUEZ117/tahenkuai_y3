def main():
    args = parse_args() # 解析输入参数。
    if args.category_num <= 0: # 类别错误。
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model): # .trt文件路径错误。
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args) # 根据args(不同输入源)读取图像并返回cam实例。
    if not cam.isOpened(): # 正常start之后is_opened应为True。
        raise SystemExit('ERROR: failed to open camera!') # 否则视为开启失败。

    cls_dict = get_cls_dict(args.category_num) # 根据类别数返回类名称字典集合。
    yolo_dim = args.model.split('-')[-1] # 分离yolo对应的模型尺寸参数。
    if 'x' in yolo_dim: # 模型尺寸参数中带有WxH(e.g.416x256)的情况。
        dim_split = yolo_dim.split('x') # 分离w和h。
        if len(dim_split) != 2: # 不是w和h两个维度则报错。
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1]) # 提取w和h。
    else: # 模型尺寸参数中W和H相同的情况。
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0: # 模型尺寸参数中h和w都要是32的倍数，否则报错。
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num) # 封装运行TensorRT所需的参数并返回TRT YOLO实例。

    open_window(WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height) # 打开显示窗口，命名并设置窗口大小(默认640*480)。
    vis = BBoxVisualization(cls_dict) # 画b-boxes。
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis) # 持续捕获图像并做检测。

    cam.release() # stop，thread_running和is_opened置False。
    cv2.destroyAllWindows() # 删除所有窗口。


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False # 默认非全屏。
    fps = 0.0 # 帧率。
    tic = time.time() # 返回当前时间的时间戳。
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0: # 获取窗口属性，关闭窗口时退出程序。
            break
        img = cam.read() # 从camera结构体读取一帧图像。
        if img is None: # camera runs out of image or error。
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th) # 检测目标，包括preprocess和postprocess。
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps) # Draw fps number at top-left corner of the image。
        cv2.imshow(WINDOW_NAME, img) # 显示。
        toc = time.time() # 返回当前时间的时间戳。
        curr_fps = 1.0 / (toc - tic) # 计算当前帧率。
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1) # 等待键盘输入。
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn) # 切换全屏。

class TrtYOLO(object):
    """TrtYOLO class encapsulates things needed to run TRT YOLO."""

    def _load_engine(self):
        TRTbin = 'yolo/%s.trt' % self.model # 模型对应的文件名。
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime: # 从.trt文件中读取engine并反序列化来执行推断(需要创建一个runtime对象)。
            return runtime.deserialize_cuda_engine(f.read()) # 返回反序列化后的结果，optimized ICudaEngine for executing inference on a built network。

    def __init__(self, model, input_shape, category_num=80, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.category_num = category_num
        self.cuda_ctx = cuda_ctx # 默认CUDA上下文只能从创建它的CPU线程访问，其他线程访问需push/pop从创建它的线程中弹出它，这样context可以被推送到任何其他CPU线程的当前上下文栈，并且随后的CUDA调用将引用该上下文。
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.inference_fn = do_inference if trt.__version__[0] < '7' \
                                         else do_inference_v2 # 不同TensorRT版本。
        self.trt_logger = trt.Logger(trt.Logger.INFO) # 打印日志，启动一个logging界面，抑制warning和errors，仅报告informational messages。
        self.engine = self._load_engine() # 加载TRT引擎并执行反序列化。

        try: # 创建一个上下文，储存中间值，因为engine包含network定义和训练参数，因此需要额外的空间。
            self.context = self.engine.create_execution_context() # create_execution_context是写在ICudaEngine.py的一个闭源方法，这个方法是创建立一个IExecutionContext类型的对象。
            grid_sizes = get_yolo_grid_sizes(
                self.model, self.input_shape[0], self.input_shape[1]) # 获取yolo网格大小，tiny模型只有1/32和1/16两个scale。
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine, grid_sizes) # 为输入输出分配host和device的buffers。
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def detect(self, img, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = _preprocess_yolo(img, self.input_shape) # 预处理，原始图像(numpy array)由int8(h,w,3)根据input_shape转换成float32(3,H,W)。

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(img_resized) # 图片转换为内存连续存储的数组(运行速度更快)，给input赋值(设置inputs[0]中的host，即input的host_mem)。
        if self.cuda_ctx: # 默认None。
            self.cuda_ctx.push()
        trt_outputs = self.inference_fn( # 进行推理，TensorRT7版本以上，调用do_inference_v2。
            context=self.context, # context是用来执行推断的对象，初始化时通过engine.create_execution_context()生成。
            bindings=self.bindings, # bindings中存的是每个input/output所占byte数的int值。
            inputs=self.inputs, # 由一个个HostDeviceMem类型组成的list，比如inputs[0]就在之前的步骤被赋值为预处理后的image。
            outputs=self.outputs, # 由一个个HostDeviceMem类型组成的list，outputs在没有执行推断之前，值为0。返回对应三个yolo scale的三个HostDeviceMem对象。
            stream=self.stream) # stream为在allocate_buffers中由cuda.Stream()生成的stream，来自于Stream.py,但是这个不是TensorRT的东西，而来自于pycuda，是cuda使用过程不可缺少的一员。
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        boxes, scores, classes = _postprocess_yolo(
            trt_outputs, img.shape[1], img.shape[0], conf_th) # 后处理。trt_outputs: a list of 2 or 3 tensors, where each tensor contains a multiple of 7 float32 numbers in the order of [x, y, w, h, box_confidence, class_id, class_prob]。

        # clip x1, y1, x2, y2 within original image
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img.shape[1]-1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img.shape[0]-1)
        return boxes, scores, classes

def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs] # 把input中的数据从主机内存复制到设备内存(递给GPU)，而inputs中的元素恰好是函数可以接受的HostDeviceMem类型。
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle) # 利用GPU执行推断的步骤，异步。
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs] # 把计算完的数据从device(GPU)拷回host memory中。
    # Synchronize the stream
    stream.synchronize() # 同步。
    # Return only the host outputs.
    return [out.host for out in outputs] # 把存在HostDeviceMem类型的outpus中的host中的数据，取出来，放在一个list中返回。