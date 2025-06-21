# pip install torch torchvision torchaudio tqdm opencv-python onnxruntime
# pip install torch torchvision torchaudio tqdm opencv-python onnxruntime-gpu

# https://github.com/DAI-Lab/RivaGAN/tree/master
# https://arxiv.org/abs/1909.01285
# https://deepwiki.com/DAI-Lab/RivaGAN?tab=readme-ov-file
# https://onnxruntime.ai/

import onnxruntime as ort  
import cv2  
import numpy as np  
import torch  
from tqdm import tqdm  
import random


def generate_random_tuple(size):
    return tuple(random.randint(0, 1) for _ in range(size))

# def compare_tuples(tuple1, tuple2):
#     if len(tuple1) != len(tuple2):
#         return "Кортежи разной длины"
#     return [a == b for a, b in zip(tuple1, tuple2)]

def compare_tuples(tuple1, tuple2):
    if len(tuple1) != len(tuple2):
        return "Кортежи разной длины"

    comparison_results = [a == b for a, b in zip(tuple1, tuple2)]
    
    count_true = comparison_results.count(True)
    count_false = comparison_results.count(False)
    false_indices = [index for index, value in enumerate(comparison_results) if not value]

    if count_false == 0:
        return count_true, "Нет значений False"
    
    return count_true, count_false, false_indices

  
class RivaGANONNX:  
    def __init__(self, encoder_path, decoder_path, data_dim=32):  
        self.data_dim = data_dim  
        self.encoder_session = ort.InferenceSession(encoder_path)  
        self.decoder_session = ort.InferenceSession(decoder_path)  
          
        # Получаем имена входных и выходных тензоров  
        self.encoder_input_names = [inp.name for inp in self.encoder_session.get_inputs()]  
        self.encoder_output_names = [out.name for out in self.encoder_session.get_outputs()]  
        self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]  
        self.decoder_output_names = [out.name for out in self.decoder_session.get_outputs()]  
      
    def encode(self, video_in, data, video_out):  
        assert len(data) == self.data_dim  
          
        video_in = cv2.VideoCapture(video_in)  
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))  
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))  
        length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))  
          
        # Подготавливаем данные как в оригинале  
        data_array = np.array([data], dtype=np.float32)  
          
        video_out = cv2.VideoWriter(  
            video_out, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))  
          
        for i in tqdm(range(length)):  
            ok, frame = video_in.read()  
            if not ok:  
                break  
                  
            # Нормализация как в оригинале: / 127.5 - 1.0  
            frame = frame.astype(np.float32) / 127.5 - 1.0  
            # Перестановка размерностей: (H, W, 3) -> (1, 3, 1, H, W)  
            frame = np.transpose(frame, (2, 0, 1))  # (3, H, W)  
            frame = np.expand_dims(frame, axis=0)   # (1, 3, H, W)  
            frame = np.expand_dims(frame, axis=2)   # (1, 3, 1, H, W)  
              
            # ONNX inference  
            encoder_inputs = {  
                self.encoder_input_names[0]: frame,  
                self.encoder_input_names[1]: data_array  
            }  
            wm_frame = self.encoder_session.run(self.encoder_output_names, encoder_inputs)[0]  
              
            # Обрезка значений как в оригинале  
            wm_frame = np.clip(wm_frame, -1.0, 1.0)  
              
            # Конвертация обратно в uint8 как в оригинале  
            wm_frame = wm_frame[0, :, 0, :, :]  # (3, H, W)  
            wm_frame = np.transpose(wm_frame, (1, 2, 0))  # (H, W, 3)  
            wm_frame = ((wm_frame + 1.0) * 127.5).astype(np.uint8)  
              
            video_out.write(wm_frame)  
          
        video_out.release()  
        video_in.release()  
      
    def decode(self, video_in):  
        video_in = cv2.VideoCapture(video_in)  
        length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))  
          
        for i in tqdm(range(length)):  
            ok, frame = video_in.read()  
            if not ok:  
                break  
                  
            # Нормализация как в оригинале  
            frame = frame.astype(np.float32) / 127.5 - 1.0  
            # Перестановка размерностей: (H, W, 3) -> (1, 3, 1, H, W)  
            frame = np.transpose(frame, (2, 0, 1))  
            frame = np.expand_dims(frame, axis=0)  
            frame = np.expand_dims(frame, axis=2)  
              
            # ONNX inference  
            decoder_inputs = {self.decoder_input_names[0]: frame}  
            data = self.decoder_session.run(self.decoder_output_names, decoder_inputs)[0]  
              
            yield data[0]  # Возвращаем первый элемент батча  
          
        video_in.release()


# Инициализация  
model = RivaGANONNX("rivagan_encoder.onnx", "rivagan_decoder.onnx")  
  
# Кодирование  
size = 32  # Размер кортежа
data = generate_random_tuple(size)
print(data)
# ffmpeg -i 4.mp4 -c copy 4.avi
invideo = "4.avi"
outvideo = "output.avi"
model.encode(invideo, data, outvideo)


# data = tuple([0] * 32)  # 32-битное сообщение  # out = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1)
# model.encode("input.avi", data, "output.avi")  
   
# Декодирование
recovered_data = list(model.decode(outvideo))

# Преобразование списка массивов в массив NumPy
data_array = np.array(recovered_data)

# Вычисление среднего значения по всем массивам
mean_values = np.mean(data_array, axis=0)

# Преобразование в кортеж
decoded_message = tuple(mean_values)

# Теперь decoded_message содержит усредненные значения в виде одного кортежа
# print(decoded_message)

# Преобразование в кортеж 0 и 1
threshold = 0.5  # Пороговое значение
binary_tuple = tuple(1 if value > threshold else 0 for value in decoded_message)

# Вывод результата
print(binary_tuple)

result = compare_tuples(data, binary_tuple)

if isinstance(result, str):
    print(result)  # Сообщение об ошибке, если кортежи разной длины
else:
    count_true = result[0]
    if len(result) == 2:
        count_false_or_message = result[1]
        print(f"Количество True: {count_true}")
        print(count_false_or_message)  # Сообщение о том, что нет значений False
    else:
        count_false = result[1]
        false_indices = result[2]
        print(f"Количество True: {count_true}")
        print(f"Количество False: {count_false}")
        print(f"Индексы False: {false_indices}")

# (0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0)
#  50%|███████████████████████████████████████████████████████████████████▉                                                                    | 393/787 [03:37<03:38,  1.81it/s]
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 393/393 [04:21<00:00,  1.51it/s]
# (0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0)
# Количество True: 32
# Нет значений False