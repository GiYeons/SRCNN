ASCNN_37.14917319500478

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
