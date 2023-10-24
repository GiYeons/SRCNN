# SRCNN

# Quantization
Quantization-related files:

1) quantized_predict.py
2) ASFSR_final.py
   
File 1. is the inference code, and 2. is the model code for ASCNN. In 1., the saved weights are loaded into the model, and the model.quantize() function is executed. This function quantizes the loaded weights. Quantization-related functions are written below the #----quantization---- comment in file 2.

Most of the functions were written by extracting only the necessary parts from the original code. ~~However, the code for applying bit shift right after convolution has not been written yet.~~ Also, the code is still in the process of being cleaned up.


preset: self.wts_nbit, self.wts_fbit = 8, 4
        self.biases_nbit, self.biases_ibit = 8, 4
        self.act_nbit, self.act_fbit = 8, 4
        self.scales_nbit, self.scales_ibit = 1, 1
PSNR of Set5: 28.75655 (bias quantize&constrain X)
Current output images (Set5):

![baby](https://github.com/GiYeons/SRCNN/assets/65033360/b59f377b-90f2-4e94-a112-111232284826)
![bird](https://github.com/GiYeons/SRCNN/assets/65033360/6b5174db-52d0-4941-9b7f-9a141b09fe1f)
