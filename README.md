# SRCNN

# Quantization
Quantization-related files:

1) quantized_predict.py
2) ASFSR_final.py
   
File 1. is the inference code, and 2. is the model code for ASCNN. In 1., the saved weights are loaded into the model, and the model.quantize() function is executed. This function quantizes the loaded weights. Quantization-related functions are written below the #----quantization---- comment in file 2.

Most of the functions were written by extracting only the necessary parts from the original code. However, the code for applying bit shift right after convolution has not been written yet. Also, the code is still in the process of being cleaned up.
