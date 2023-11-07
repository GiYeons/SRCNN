# SRCNN

# Quantization
Quantization-related files:

1) quantized_predict.py
2) ASFSR_final.py
   
File 1. is the inference code, and 2. is the model code for ASCNN. In 1., the saved weights are loaded into the model, and the model.quantize() function is executed. This function quantizes the loaded weights. Quantization-related functions are written below the #----quantization---- comment in file 2(line 187).

The quantize() function runs the prepare_q_weight() function; the actual quantization work is done in the prepare_q_weight() function, and the quantize() function is only responsible for storing the quantized weights in the model.

Most of the functions were written by extracting only the necessary parts from the original code. Also, the code is still in the process of being cleaned up.


preset:

        self.wts_nbit, self.wts_fbit = 8, 4
        
        self.biases_nbit, self.biases_ibit = 8, 4
        
        self.act_nbit, self.act_fbit = 8, 4
        
        self.scales_nbit, self.scales_ibit = 1, 1

  
Original PSNR of Set5: 37.09

Current PSNR of Set5: 15.81


Current output images (Set5):

![bird](https://github.com/GiYeons/SRCNN/assets/65033360/3b37146f-16ef-479a-a9d2-9ed42072494e)

