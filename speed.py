import torch
import time

from src.signal_model import WaveRNN

torch.set_grad_enabled(False)
bits = 16
local_length = 80
local_features_size = 128
upsample_convs = [4]
upsample_repeat = 75
signal_length = local_length * upsample_convs[0] * upsample_repeat
model = WaveRNN(
    hidden_size=896,
    bits=bits,
    upsample_convs=upsample_convs,
    upsample_repeat=upsample_repeat,
    local_features_size=local_features_size).eval()
local_features = torch.FloatTensor(local_length, local_features_size).cuda()

for device in [torch.device('cpu'), torch.device('cuda')]:
    for infer_name, infer in [('model.infer_2', model.infer_2), ('model.infer', model.infer)]:
        model = model.to(device)
        local_features = local_features.to(device)

        print('Infer:', infer_name)
        print('Device:', device)

        start = time.time()
        predicted_coarse, predicted_fine, _ = infer(local_features)

        print('Performance:', signal_length / (time.time() - start), 'it/s')
        print('-' * 10)
