import argparse
import os
import numpy as np
import time

from model import CCGAN
from preprocess import *

def conversion(speaker_name_list, src, tar, model_dir, model_name, data_dir, output_dir):
    
    start_time = time.time()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    out_dir = os.path.join(output_dir, 'A=' + src + '_B=' + tar)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_features = 24
    sampling_rate = 16000
    frame_period = 5.0
    
    num_speakers = len(speaker_name_list)
    
    src_idx = speaker_name_list.index(src)
    tar_idx = speaker_name_list.index(tar)
    
    src_id = [0.]*num_speakers
    src_id[src_idx] = 1.0
    
    tar_id = [0.]*num_speakers
    tar_id[tar_idx] = 1.0

    model = CCGAN(num_features = num_features, num_speakers = num_speakers, mode = 'test')

    model.load(filepath = os.path.join(model_dir, model_name))
    
    src_npload = np.load(os.path.join(model_dir, src + '.npz'), allow_pickle = True)
    src_sps_mean = src_npload['sps_mean']
    src_sps_std = src_npload['sps_std']
    src_f0s_mean = src_npload['f0s_mean']
    src_f0s_std = src_npload['f0s_std']
    
    tar_npload = np.load(os.path.join(model_dir, tar + '.npz'), allow_pickle = True)
    tar_sps_mean = tar_npload['sps_mean']
    tar_sps_std = tar_npload['sps_std']
    tar_f0s_mean = tar_npload['f0s_mean']
    tar_f0s_std = tar_npload['f0s_std']
    
    data_dir = os.path.join(data_dir, src)
    assert os.path.exists(data_dir)

    for file in os.listdir(data_dir):
        filepath = os.path.join(data_dir, file)
        wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
        wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_features)
        coded_sp_transposed = coded_sp.T

        f0_converted = pitch_conversion(f0 = f0, mean_log_src = src_f0s_mean, std_log_src = src_f0s_std, mean_log_target = tar_f0s_mean, std_log_target = tar_f0s_std)
        coded_sp_norm = (coded_sp_transposed - src_sps_mean) / src_sps_std
        coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), A_id = src_id, B_id = tar_id)[0]
        coded_sp_converted = coded_sp_converted_norm * tar_sps_std + tar_sps_mean
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
        librosa.output.write_wav(os.path.join(out_dir, src + '_to_' + tar + '_' + os.path.basename(file)), wav_transformed, sampling_rate)
    
    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Conversion Done.')
    print('Time Elapsed for conversion: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert voices using pre-trained CCGAN model.')
    
    speaker_name_list_default = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 'TF1', 'TF2', 'TM1', 'TM2', 'TM3']
    src_default = 'SF1'
    tar_default = 'TM3'
    model_dir_default = './model/'
    model_name_default = 'CCGAN.ckpt'
    data_dir_default = './data/evaluation_all/'
    output_dir_default = './converted_voices/'
    
    parser.add_argument('--speaker_name_list', type = str, nargs='+', help = 'Full Speaker Name List who participated in CCGAN training.', default = speaker_name_list_default)
    parser.add_argument('--src', type = str, help = 'Source Speaker.', default = src_default)
    parser.add_argument('--tar', type = str, help = 'Target Speaker.', default = tar_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for the pre-trained model.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'Filename for the pre-trained model.', default = model_name_default)
    parser.add_argument('--data_dir', type = str, help = 'Directory for the voices for conversion.', default = data_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for the converted voices.', default = output_dir_default)
    
    argv = parser.parse_args()
    
    speaker_name_list = argv.speaker_name_list
    src = argv.src
    tar = argv.tar
    model_dir = argv.model_dir
    model_name = argv.model_name
    data_dir = argv.data_dir
    output_dir = argv.output_dir

    conversion(speaker_name_list = speaker_name_list, src = src, tar = tar, model_dir = model_dir, model_name = model_name, data_dir = data_dir, output_dir = output_dir)


