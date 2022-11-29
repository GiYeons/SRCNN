# with h5py.File("dataset/train_mscale.h5") as file:
#     f = file['label']
#     image = f[0]
#     image = np.moveaxis(image, 0, -1)
#     print(image.shape)
#     plt.imshow(image, interpolation='none')
#     plt.show()

# 데이터 시각화
# for x, y in train_dataloader:
#     print(x.shape, y.shape)
#     break
# img = x[0]
# target = y[0]
# print(img.shape, target.shape)
# print(img.dtype)
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(to_pil_image(img), cmap='gray')
# plt.title('train')
# plt.subplot(1, 2, 2)
# plt.imshow(to_pil_image(target), cmap='gray')
# plt.title('target')
# plt.show()

# butterfly로 PSNR 검증
# img = imread("dataset/butterfly.bmp")
#     img = rgb2ycbcr(img)[:, :, 0:1]/255
#     h = img.shape[1]
#     img = img[0:h - h%3, 0:h - h%3, :]
#     smallImg = imresize(img, scalar_scale=1/3, method='bicubic')
#     bicImg = imresize(smallImg, scalar_scale=3, method='bicubic')
#     # smallImg = F.interpolate(img, scale_factor=1/3, mode='bicubic')
#     # bicImg = F.interpolate(smallImg, scale_factor=3, mode='bicubic')
#
#     print(PSNR(img, bicImg))
#
#     plt.subplot(1, 3, 1)
#     plt.title("label")
#     plt.imshow(img, cmap='gray')
#     plt.subplot(1, 3, 2)
#     plt.title("downsampled")
#     plt.imshow(smallImg, cmap='gray')
#     plt.subplot(1, 3, 3)
#     plt.title("bicubic_x3")
#     plt.imshow(bicImg, cmap='gray')
#     plt.show()

# val 데이터셋 준비

# images = sorted(os.listdir(val_path))
# print(images[0])
# for image in images:
#     img = imread(val_path + image)
#     img = rgb2ycbcr(img)[:, :, 0:1]     # 왜 BGR 배열을 그대로 집어넣어도 이 함수가 작동하는지는 모르겠다
#     height, width, channel = img.shape
#
#     label = img[0:height - (height % scale), 0: height - (width % scale), :]
#     lr = imresize(label, scalar_scale=1/scale, method='bicubic')

# validate 함수
# with torch.no_grad():
#     for bi, data in enumerate(tqdm(images)):
#         image_data = data[0].to(device)  # LR
#         label = data[1].to(device)  # HR
#
#         outputs = model(image_data, border_cut)
#         label = label[:, :, border_cut:-border_cut, border_cut:-border_cut]
#         loss = criterion(outputs, label)
#
#         running_loss += loss.item()  # batch loss
#         batch_psnr = PSNR(label, outputs)
#         running_psnr += batch_psnr
#     outputs = outputs.cpu()
#     save_image(outputs, f"outputs/val_sr{epoch}.png")
#
# final_loss = running_loss / len(dataloader.dataset)
# final_psnr = running_psnr / (int(len(dataloader.dataset) / dataloader.batch_size))
# return final_loss, final_psnr

# 데이터셋  getitem
# def __getitem__(self, index):
#     self.dataset = h5py.File(self.path, 'r')
#     self.lr = self.dataset['x_set'][str(index + 1)]
#     self.label = self.dataset['y_set'][str(index + 1)]
#     return (
#         torch.tensor(np.array(self.lr), dtype=torch.float32) / 255.0,
#         torch.tensor(np.array(self.label), dtype=torch.float32) / 255.0
#     )

# dataset = TrainDataset(train_path)
# dataset_size = len(dataset)  # 896573 members
# train_size = 700000
# not_used_size = dataset_size - train_size  # 데이터가 너무 크니까 일부만 사용하려고 분리하는것
# # print(train_size+not_used_size)
# train_data, not_used_data = random_split(dataset, [train_size, not_used_size])  # dataset을 split


# psnr = 0
# list = ["baby", "bird", "butterfly", "head", "woman"]
# for item in list:
#     img = imread("dataset/%s.bmp" %(item))
#     img = rgb2ycbcr(img)[:, :, 0:1] / 255
#     # img = img[2:-2, 2:-2, :]
#     print(img.shape)
#
#     w = img.shape[0]
#     h = img.shape[1]
#     img = img[0:w - w % 2, 0:h - h % 2, :]
#
#     smallImg = imresize(img, scalar_scale=1 / 2, method='bicubic')
#     bicImg = imresize(smallImg, scalar_scale=2, method='bicubic')
#     # smallImg = F.interpolate(img, scale_factor=1/2, mode='bicubic')
#     # bicImg = F.interpolate(smallImg, scale_factor=2, mode='bicubic')
#
#     print(testPSNR(img, bicImg))
#     psnr += testPSNR(img, bicImg)
#
# print("average:", psnr/5)
# 33.662560258929304
# SRCNN 논문: 33.66


## transposed convolution 테스트
# img = torch.randn(1 ,50 ,28 ,28)
# kernel = torch.randn(30,50 ,3 ,3)
# true_convt2d = F.conv_transpose2d(img, kernel.transpose(0,1))
#
# pad0 = 3-1 # to explicitly show calculation of convtranspose2d padding
# pad1 = 3-1
# inp_unf = torch.nn.functional.unfold(img, (3,3), padding=(pad0,pad1))
# print(inp_unf.shape, inp_unf[:, :1, :50])
# w = torch.rot90(kernel, 2, [2,3])
# # this is done the same way as forward convolution
# out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
# out = out_unf.view(true_convt2d.shape)
# print((true_convt2d-out).abs().max())
# print(true_convt2d.abs().max())


# t = torch.arange(0, 50).reshape((2,1,5,5))
# par = torch.arange(0,9).reshape((3,3))
# print(t)
# t3 = t.unfold(2,3,1).unfold(3,3,1)
# # print("UNFOLD1\n", t2)
# # t3 = t.
# print("UNFOLD2\n", t3)
# print(t3.shape)
# t3 = t3.transpose(0,1)
# print("transpose: ", t3.shape, t3)
#
# t3 = t3.contiguous().view(1,-1, 3,3)
# print("patches", t3)
# t3 = t3.transpose(0,1)
# print("last transpose", t3.shape, t3)
# print(t3[0])


# t = torch.tensor([[
#     [[1,2],
#     [3,4,]]]])
# print(t)
# t = t.repeat(1,4,1,1)
# print(t)
# t = F.pixel_shuffle(t, 2)
# print(t)


############## decision mask test #############
# plt.subplot(1,2,1)
# imshow(img, cmap='gray')
# img = np.moveaxis(img, 2, 0)
# img = torch.tensor(img).float() / 255.
# blur = F.avg_pool2d(img, kernel_size=3, stride=1, padding=3//2, count_include_pad=False)
# mask = torch.where(torch.abs(img-blur) >= 0.04, 1, 0).float()
# mask = F.max_pool2d(mask.float(), kernel_size=3, padding=3//2)
#
# mask = np.moveaxis(mask.numpy(), 0, 2)
# plt.subplot(1,2,2)
# imshow(mask)
# plt.show()

# model = Net(2)
# outputs = model(img, 0.04, 3)
# print(outputs)


################ convTranspose2d test ####################
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.trans2d = nn.ConvTranspose2d(
#             in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2, bias=False)
#
#         self.trans2d.weight.data = nn.Parameter(torch.tensor([[[[0.1,0.2,0.3], [0.4,0.5,0.6]]]], dtype=torch.float32))
#
#     def forward(self, x):
#         y = self.trans2d(x)
#         return y
#
#
# t = torch.tensor([[[[1,2,3], [4,5,6]]]], dtype=torch.float32)
# model = Net()
# for param in model.parameters():
#     print(param)
#
# output = model(t)
# print(output)