import torch.onnx
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType
import cv2
from imgaug.augmenters import Resize
from torchvision import transforms
from laneatt import LaneATT
import numpy as np
import os
import time
from torch.quantization import quantize_fx
from torch.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

def to_onnx():
    resize = Resize({'height': 360, 'width': 640})
    to_tensor = transforms.ToTensor()
    result = torch.load('torch_output.pt')
    result = to_numpy(result)

    img = cv2.imread('camera_raw2.jpg')
    img = resize(image=img)
    img = to_tensor(img)
    img = img.unsqueeze(0)  # batch size 1

    model = LaneATT(anchors_freq_path='', topk_anchors=1000)

    state = torch.load('')
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    # out = model(img)
    # out = to_numpy(out)
    # print(np.array_equal(result, out))
    # if not np.array_equal(result, out):
    #     # test = np.equal(out, result)
    #     # idx = np.argwhere(test)
    #     # print((result[0, idx] - out[0, idx]).sum())
    #     np.testing.assert_allclose(result, out)

    torch.onnx.export(model, img, f='laneatt2.onnx', export_params=True, input_names=['image'], output_names=['reg_proposals'],
                      opset_version=15, do_constant_folding=True)


def check_onnx():
    onnx_model = onnx.load("laneatt2.onnx")
    onnx.checker.check_model(onnx_model)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def test_ort():
    resize = Resize({'height': 360, 'width': 640})
    to_tensor = transforms.ToTensor()
    result = torch.load('torch_output.pt')
    result = to_numpy(result)

    img = cv2.imread('camera_raw2.jpg')
    img = resize(image=img)
    img = to_tensor(img)
    img = img.unsqueeze(0)  # batch size 1
    img = to_numpy(img)

    ort_session = ort.InferenceSession("quantized_laneatt_S8S8.onnx", providers=['CPUExecutionProvider'])
    ort_inputs = {'image' : img}
    ort_outs = ort_session.run(['reg_proposals'], ort_inputs)

    # laneatt = LaneATT(anchors_freq_path='', topk_anchors=1000)
    #
    # out = laneatt.nms(torch.Tensor(ort_outs[0]), nms_thresh=50., nms_topk=4, conf_threshold=.5, device='cpu')
    # result = laneatt.nms(result, nms_thresh=50., nms_topk=4, conf_threshold=.5, device='cpu')
    #
    # out = laneatt.decode(out, as_lanes=True)
    # result = laneatt.decode(result, as_lanes=True)
    #
    # print(out)
    # print('----------------------')
    # print(result)

    err = np.abs(result - ort_outs[0])
    err_c = err[0, :, :2]
    err_s = err[0, :, 2]
    err_l = err[0, :, 4]
    err_x = err[0, :, 5:]

    print('Avg:', err_c.mean(), err_s.mean(), err_l.mean(), err_x.mean())
    print('Median:', np.median(err_c), np.median(err_s), np.median(err_l), np.median(err_x))
    print('Std:', err_c.std(), err_s.std(), err_l.std(), err_x.std())
    print('Max:', err_c.max(), err_s.max(), err_l.max(), err_x.max())
    print('-------------------')
    print('Quantiles:')
    print('.75:', np.quantile(err_c, .75), np.quantile(err_s, .75), np.quantile(err_l, .75), np.quantile(err_x, .75))
    print('.9:', np.quantile(err_c, .9), np.quantile(err_s, .9), np.quantile(err_l, .9), np.quantile(err_x, .9))
    print('.99:', np.quantile(err_c, .99), np.quantile(err_s, .99), np.quantile(err_l, .99), np.quantile(err_x, .99))
    # np.testing.assert_allclose(result, ort_outs[0], rtol=1e-3, atol=0)
    # print('qlaneatt successful')


class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            data_list = preprocess_func(self.image_folder)
            self.datasize = len(data_list)
            print('num imgs:', self.datasize)
            self.enum_data_dicts = iter([{'image': data} for data in data_list])
        return next(self.enum_data_dicts, None)


def preprocess_func(list):
    '''
    Loads a batch of images and preprocess them
    return: list of matrices characterizing multiple images
    '''
    root = ''  # data path
    resize = Resize({'height': 360, 'width': 640})
    to_tensor = transforms.ToTensor()

    with open(os.path.join(root, list), 'r') as list_file:
        batch_filenames = [line.rstrip()[1 if line[0] == '/' else 0::]
                 for line in list_file]  # remove `/` from beginning if needed

    batch_filenames = [os.path.join(root, file) for file in batch_filenames[:50]]
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        img = cv2.imread(image_name)
        img = resize(image=img)
        img = to_tensor(img)
        img = img.unsqueeze(0)  # batch size 1 (.unsqueeze(0) again for tensor output)
        img = to_numpy(img)
        unconcatenated_batch_data.append(img)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)  # have to expand twice so when iterating through, returns batch_like img
    # batch_data = torch.cat(unconcatenated_batch_data, dim=0)
    return batch_data


def quantize_onnx():
    data_reader = DataReader('list/test_small.txt')

    ort.quantization.quantize_static('laneatt2.onnx', 'quantized_laneatt_S8S8.onnx', calibration_data_reader=data_reader,
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    reduce_range=False, per_channel=False, optimize_model=True, extra_options={'ActivationSymmetric' : False, 'WeightSymmetric' : True})


def time_onnx(model_path, ort_inputs):
    so = ort.SessionOptions()
    ort_session = ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])

    # warmup
    for _ in range(10):
        i = ort_session.run(['reg_proposals'], ort_inputs)

    # time
    total = 0
    for _ in range(100):
        t1 = time.time()
        i = ort_session.run(['reg_proposals'], ort_inputs)
        t2 = time.time()
        total += t2 - t1
    print(model_path, 'throughput:')
    print('Avg iter/s: ', 100 / total)


def compare():
    resize = Resize({'height': 360, 'width': 640})
    to_tensor = transforms.ToTensor()

    img = cv2.imread('camera_raw2.jpg')
    img = resize(image=img)
    img = to_tensor(img)
    img = img.unsqueeze(0)  # batch size 1
    img = to_numpy(img)
    ort_inputs = {'image': img}

    # time_onnx('laneatt.onnx', ort_inputs)
    # time_onnx('laneatt-opt.onnx', ort_inputs)
    time_onnx('laneatt2.onnx', ort_inputs)
    time_onnx('laneatt2-opt.onnx', ort_inputs)
    # time_onnx('quantized_laneatt_S8S8.onnx', ort_inputs)


def pt_quant():
    model = LaneATT(anchors_freq_path='', topk_anchors=1000) # data path

    state = torch.load('') # data path
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    qconfig = torch.quantization.QConfig(
        activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(ch_axis=0, dtype=torch.qint8)
    )
    qconfig_dict = {"": qconfig}
    # Prepare
    model_prepared = quantize_fx.prepare_fx(model.feature_extractor, qconfig_dict)
    # Calibrate - Use representative (validation) data.

    data = preprocess_func('list/test_small.txt')
    with torch.inference_mode():
        for x in data:
            model_prepared(x)
    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    print(model_quantized(data[0]).shape)
    print(type(model_quantized))
    torch.save(model_quantized.state_dict(), 'quantized_resnet18.pt')


def test_qres():
    model = LaneATT(anchors_freq_path='', topk_anchors=1000)  # data path
    model.eval()

    qconfig = torch.quantization.QConfig(
        activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(ch_axis=0, dtype=torch.qint8)
    )
    qconfig_dict = {"": qconfig}
    # Prepare
    qres = quantize_fx.prepare_fx(model.feature_extractor, qconfig_dict)
    qres = quantize_fx.convert_fx(qres)

    state = torch.load('')
    qres.load_state_dict(torch.load(''))
    model.load_state_dict(state['model'], strict=True)
    model.feature_extractor = qres

    resize = Resize({'height': 360, 'width': 640})
    to_tensor = transforms.ToTensor()
    result = torch.load('torch_output.pt')
    result = to_numpy(result)

    img = cv2.imread('camera_raw2.jpg')
    img = resize(image=img)
    img = to_tensor(img)
    img = img.unsqueeze(0)  # batch size 1

    for _ in range(10):
        o = model(img)

    total = 0
    for _ in range(100):
        t1 = time.time()
        o = model(img)
        t2 = time.time()
        total += t2 - t1

    print('Avg:', total / 100, 100 / total)

    # out = model(img)
    # out = to_numpy(out)
    #
    # np.testing.assert_allclose(result, out)


if __name__ == "__main__":
    ## ONNX Quantization
    # to_onnx()
    # check_onnx()
    # test_ort()
    # quantize_onnx()
    # test_ort()
    compare()

    ## PyTorch Quantization
    # pt_quant()
    # test_qres()


