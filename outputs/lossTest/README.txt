loss test

    baseline.pth: real baseline. from ASCNN
    PSNR = 37.18109, Set14 = 32.7
    Tconv=
    parameters: 13777
    
    baseline2.pth: mask setting 1
    PSNR = 37.20854, Set14 = 32.75866
    Tconv=
    parameters: 13777
    
    my_baseline.pth: training with ASCNN baseline and my train.py
    PSNR = 37.22, Set14 = 32.73
    PSNR_specific = 38.21
    parameters: 13777

    loss0.9.pth: 
    PSNR = 37.23, Set14 = 32.73
    Tconv=
    parameters: 13809

    ##### w = 2.3 ####
    sparse_loss11.pth: th = 0.03, 
    PSNR = 37.12, Set14 = 32.65
    PSNR_specific = 38.11563
    parameters: 13809
    
    ##### w = 1.1 ####
    sparse_loss21.pth: th = 0.04, 
    PSNR = 37.20, Set14 = 32.76
    PSNR_specific = 38.19
    parameters: 13809
    
    ##### w = 0.8 ####
    sparse_loss31.pth: th = 0.04, 
    PSNR = 37.19, Set14 = 32.72
    PSNR_specific = 38.17
    parameters: 13809
    
    ##### w = 0.9 ####
    sparse_loss41.pth: th = 0.03, 
    PSNR = 37.22, Set14 = 32.75
    PSNR_specific = 38.21
    parameters: 13809



