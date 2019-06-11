import os
import numpy as np
import argparse
import time
import librosa

from preprocess import *
from model import CCGAN


def train(speaker_name_list, num_epochs, train_dir, validation_dir, model_dir, model_name, random_seed, output_dir, tensorboard_log_dir):
    
    print(speaker_name_list)
    
    np.random.seed(random_seed)
    num_speakers = len(speaker_name_list)
    
    num_sentences = 162 # min(number of setences for each speaker)
    num_epochs = num_epochs
    no_decay = num_epochs * num_sentences * (num_speakers ** 2) / 2.0
    id_mappig_loss_zero = 10000 * (num_speakers ** 2) # can be less
    
    mini_batch_size = 1 # must be 1
    generator_learning_rate = 0.0002
    
    generator_learning_rate_decay = generator_learning_rate / no_decay
    discriminator_learning_rate = 0.0001
    discriminator_learning_rate_decay = discriminator_learning_rate / no_decay
    sampling_rate = 16000
    num_mcep = 24
    frame_period = 5.0
    n_frames = 128
    
    lambda_cycle = 10
    lambda_identity = 5
    lambda_A2B = 3 # 1 is bad, 3~8 is good
    
    dic_sps_norm = {}
    dic_sps_mean = {}
    dic_sps_std = {}
    dic_f0s_mean = {}
    dic_f0s_std = {}

    print('Preprocessing Data...')

    start_time = time.time()
    
    for speaker in speaker_name_list:
        if not os.path.exists(os.path.join(model_dir, speaker + '.npz')):
            wavs = load_wavs(wav_dir = os.path.join(train_dir, speaker), sr = sampling_rate)
            f0s, timeaxes, sps, aps, coded_sps = world_encode_data(wavs = wavs, fs = sampling_rate, frame_period = frame_period, coded_dim = num_mcep)
            log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
            coded_sps_transposed = transpose_in_list(lst = coded_sps)
            coded_sps_norm, coded_sps_mean, coded_sps_std = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_transposed)
            np.savez(os.path.join(model_dir, speaker + '.npz'), coded_sps = coded_sps, f0s_mean = log_f0s_mean, f0s_std = log_f0s_std, sps_mean = coded_sps_mean, sps_std = coded_sps_std)
            dic_sps_mean[speaker] = coded_sps_mean
            dic_sps_std[speaker] = coded_sps_std
            dic_f0s_mean[speaker] = log_f0s_mean
            dic_f0s_std[speaker] = log_f0s_std
        else:    
            npload = np.load(os.path.join(model_dir, speaker + '.npz'), allow_pickle = True)
            coded_sps = npload['coded_sps']
            dic_sps_mean[speaker] = npload['sps_mean']
            dic_sps_std[speaker] = npload['sps_std']
            dic_f0s_mean[speaker] = npload['f0s_mean']
            dic_f0s_std[speaker] = npload['f0s_std']
            coded_sps_transposed = transpose_in_list(lst = coded_sps)
            coded_sps_norm, _, _ = coded_sps_normalization_fit_transoform(coded_sps = coded_sps_transposed)
        
        dic_sps_norm[speaker] = coded_sps_norm 
        
        print('Log Pitch', speaker)
        print('Mean: %f, Std: %f' %(dic_f0s_mean[speaker], dic_f0s_std[speaker]))
    
    coded_sps_norm_list = []
    for speaker in speaker_name_list:
        coded_sps_norm_list.append(dic_sps_norm[speaker])
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    model = CCGAN(num_features = num_mcep, num_speakers = num_speakers)

    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch) # 현재 에포크 출력

        start_time_epoch = time.time()
        n_samples, dataset_src, dataset_tar = sample_train_data_mix_all_mapping(coded_sps_norm_list, n_frames = 128)
               
        for sentence_idx in range(n_samples):
            for case_src in range(num_speakers):
                for case_tar in range(num_speakers):
            
                    A_id = [0.]*num_speakers
                    B_id = [0.]*num_speakers
                    
                    A_id[case_src] = 1.0
                    B_id[case_tar] = 1.0
                    
                    mapping_direction = num_speakers * case_src + case_tar

                    num_iterations = (num_speakers * num_speakers * n_samples * epoch) + (num_speakers * num_speakers * sentence_idx) + (num_speakers * case_src) + case_tar

                    if num_iterations > id_mappig_loss_zero:
                        lambda_identity = 0
                        
                    if num_iterations > no_decay: # iteration 이 넘어가면 선형감쇠 = 점점 조금 학습
                        generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                        discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)
                        
                    generator_loss, discriminator_loss = model.train(input_A = np.expand_dims(dataset_src[sentence_idx][mapping_direction], axis = 0), input_B = np.expand_dims(dataset_tar[sentence_idx][mapping_direction], axis = 0), lambda_cycle = lambda_cycle, lambda_identity = lambda_identity, lambda_A2B = lambda_A2B, generator_learning_rate = generator_learning_rate, discriminator_learning_rate = discriminator_learning_rate, A_id = A_id, B_id = B_id)

                    if sentence_idx == 80 or sentence_idx == 161:
                        print('Epoch: {:04d} Iteration: {:010d}, Generator Learning Rate: {:.8f}, Discriminator Learning Rate: {:.8f}'.format(epoch, num_iterations, generator_learning_rate, discriminator_learning_rate))
                        print('src: {:s}, tar: {:s}, sent: {:d}, Generator Loss : {:.10f}, Discriminator Loss : {:.10f}'.format(speaker_name_list[case_src], speaker_name_list[case_tar], sentence_idx, generator_loss, discriminator_loss))

        if epoch % 50 == 0 and epoch != 0:
            model.save(directory = model_dir, filename = model_name) #모델저장

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))
       
        # validaiton is conducted for first speaker and last speaker of the speaker list
        # if you want full mapping validation, run convert_all.py
        
        if (epoch % 100 == 0 or epoch == num_epochs - 1) and epoch != 0:
        
            A_id_v = [0.]*num_speakers
            B_id_v = [0.]*num_speakers
            
            A_id_v[0] = 1.0
            B_id_v[-1] = 1.0
            
            src_v = speaker_name_list[0]
            tar_v = speaker_name_list[-1]
            
            log_f0s_mean_A = dic_f0s_mean[src_v]
            log_f0s_std_A = dic_f0s_std[src_v]
            coded_sps_A_mean = dic_sps_mean[src_v]
            coded_sps_A_std = dic_sps_std[src_v]
            
            log_f0s_mean_B = dic_f0s_mean[tar_v]
            log_f0s_std_B = dic_f0s_std[tar_v]
            coded_sps_B_mean = dic_sps_mean[tar_v]
            coded_sps_B_std = dic_sps_std[tar_v]
            
            if validation_dir is not None:
                validation_A_dir = os.path.join(validation_dir, src_v)
               
            if validation_A_dir is not None:
                validation_A_output_dir = os.path.join(output_dir, str(epoch)+'_'+src_v+'_to_'+tar_v)
                if not os.path.exists(validation_A_output_dir):
                    os.makedirs(validation_A_output_dir)

            print('validaiton (SRC: %s, TAR: %s, epoch: %d)' %(src_v, tar_v, epoch))
            for file in os.listdir(validation_A_dir):
                    filepath = os.path.join(validation_A_dir, file)
                    wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_A, std_log_src = log_f0s_std_A, mean_log_target = log_f0s_mean_B, std_log_target = log_f0s_std_B)
                    coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                    coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]),A_id = A_id_v, B_id = B_id_v)[0]
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)
                  
            A_id_v = [0.]*num_speakers
            B_id_v = [0.]*num_speakers
            
            A_id_v[-1] = 1.0
            B_id_v[0] = 1.0
            
            src_v = speaker_name_list[-1]
            tar_v = speaker_name_list[0]
            
            log_f0s_mean_A = dic_f0s_mean[src_v]
            log_f0s_std_A = dic_f0s_std[src_v]
            coded_sps_A_mean = dic_sps_mean[src_v]
            coded_sps_A_std = dic_sps_std[src_v]
            
            log_f0s_mean_B = dic_f0s_mean[tar_v]
            log_f0s_std_B = dic_f0s_std[tar_v]
            coded_sps_B_mean = dic_sps_mean[tar_v]
            coded_sps_B_std = dic_sps_std[tar_v]
            
            if validation_dir is not None:
                validation_A_dir = os.path.join(validation_dir, src_v)
               
            if validation_A_dir is not None:
                validation_A_output_dir = os.path.join(output_dir, str(epoch)+'_'+src_v+'_to_'+tar_v)
                if not os.path.exists(validation_A_output_dir):
                    os.makedirs(validation_A_output_dir)

            print('validaiton (SRC: %s, TAR: %s, epoch: %d)' %(src_v, tar_v, epoch))
            for file in os.listdir(validation_A_dir):
                    filepath = os.path.join(validation_A_dir, file)
                    wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_A, std_log_src = log_f0s_std_A, mean_log_target = log_f0s_mean_B, std_log_target = log_f0s_std_B)
                    coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                    coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]),A_id = A_id_v, B_id = B_id_v)[0]
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
                    librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Train CCGAN model for datasets.')
    
    speaker_name_list_default = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 'TF1', 'TF2', 'TM1', 'TM2', 'TM3']
    num_epochs_default = 251
    train_dir_default = './data/vcc2016_training/'
    model_dir_default = './model'
    model_name_default = 'CCGAN.ckpt'
    random_seed_default = 0
    validation_dir_default = './data/evaluation_all/'
    output_dir_default = './validation_output'
    tensorboard_log_dir_default = './log'
    
    parser.add_argument('--speaker_name_list', type = str, nargs='+', help = 'Speaker Name List.', default = speaker_name_list_default)
    parser.add_argument('--num_epochs', type = int, help = 'Number of Epoch.', default = num_epochs_default)
    parser.add_argument('--train_dir', type = str, help = 'Directory for training.', default = train_dir_default)
    parser.add_argument('--validation_dir', type = str, help = 'If set none, no conversion would be done during the training.', default = validation_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)

    argv = parser.parse_args()

    speaker_name_list = argv.speaker_name_list
    num_epochs = argv.num_epochs
    train_dir = argv.train_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_dir = None if argv.validation_dir == 'None' or argv.validation_dir == 'none' else argv.validation_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir
    
    train(speaker_name_list, num_epochs, train_dir, validation_dir, model_dir, model_name, random_seed, output_dir, tensorboard_log_dir)      
