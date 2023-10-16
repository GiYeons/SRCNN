ASCNN_37.1481

scales = [2]
    learnig_rate = 7e-4
    batch_size = 64
    epochs = 80
    iterations = 13000
    num_worker = 10

    r = 4
    th = 0.04
    dilker = 3
    dilation = False
    eval = False
    
    val_path = '/home/wstation/Set5/'
    
    data_gen = Generator(train_path, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=(epochs*iterations), eta_min=1e-10)
    
    
    
    r=16: conv=16, Tconv=4
    
    3path.pth: th=[0.004, 0.002]
    PSNR = 36.88
    Tconv=16
    
    
    
    2path1.pth: th=0.04
    r=16
    PSNR = 36.71425429242395, Set14 = 32.52294
    TConv=16
    parameters: 15089
    
    2path2.pth: th=0.04
    r=8
    PSNR = 37.12313, Set14 = 32.68044
    TConv=8
    parameters: 16224
        
    3path1.pth: th=[0.04, 0.02]
    r=[2, 8]
    PSNR = 37.19036045715474, Set14 = 32.72849
    Tconv=
    parameters: 25449
           
    3path2.pth: th=[0.04, 0.02]
    r=[4, 16]
    PSNR = 36.88000327188267, Set14 = 32.62427
    Tconv=
    parameters: 19774
           
    3path3.pth: th=[0.04, 0.02]
    r=[4, 8]
    PSNR = 37.162474170347494, Set14 = 32.7134
    Tconv=
    parameters: 20909
           
    4path1.pth: th=[0.06, 0.04, 0.002]
    r=[2, 4, 8]
    PSNR = 37.12048527036056, Set14 = 32.70673
    Tconv=
    parameters: 30134
           
    4path2.pth: th=[0.075, 0.035, 0.013]
    r=[2, 4, 8]
    PSNR = 37.13418600997662, Set14 = 32.70673
    Tconv=
    parameters: 30134
    
    4path3.pth: th=[0.075, 0.035, 0.013]
    r=[2, 8, 16]
    PSNR = 37.04007007553937, Set14 = 
    Tconv=
    parameters: 26729
   
    4path4.pth: th=[0.06, 0.04, 0.02]
    r=[2, 8, 16]
    PSNR = 36.677171105573265, Set14 = 
    Tconv=
    parameters: 26729
    
    4path5.pth: th=[0.06, 0.04, 0.02]
    r=[4, 8, 16]
    PSNR = 36.90340138209427, Set14 = 
    Tconv=
    parameters: 22189
    
    4path6.pth: th=[0.06, 0.04, 0.02]
    r=[2, 8, 16]
    PSNR = 36.773377487042694, Set14 = 
    Tconv=
    parameters: 22189
    
    4path7.pth: th=[0.075, 0.035, 0.013]
    r=[4, 8, 16]
    PSNR = 36.95564, Set14 = 
    Tconv=
    parameters: 22189
    
    
    5path1.pth: th=[0.08, 0.06, 0.04, 0.02]
    r=[2, 4, 8, 16]
    PSNR = 36.79678349279243, Set14 = 
    Tconv=
    parameters: 31414
    
    5path2.pth: th=[0.075, 0.04, 0.02, 0.01]
    r=[2, 4, 8, 16]
    PSNR = 37.10811501925331, Set14 = 
    Tconv=
    parameters: 31414
    
    
    final.pth: transposed convolution layer -> sub pixel layer
    PSNR = 37.091681759317154
