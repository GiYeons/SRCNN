import model
import os
from skimage.io import imread
from skimage.color import rgb2ycbcr
from matlab import *
from utils import *
from torchvision.utils import save_image

if __name__=="__main__":
    scale = 3
    border_cut = 32//2//2

    val_path = '/home/wstation/Set5/'
    model_path = 'outputs_x3/model.pth'

    device = torch.device('cuda:0')
    print('Computation device: ', device)

    model = model.SRCNN().to(device)
    model.load_state_dict(torch.load(model_path))

    images = sorted(os.listdir(val_path))

    sum_psnr = 0.

    model.eval()
    with torch.no_grad():
        for image in images:
            # 전처리
            img = imread(val_path + image)
            img = rgb2ycbcr(img)[:, :, 0:1]
            height, width, channel = img.shape

            label = img[0:height - (height % scale), 0: width - (width % scale), :]
            lr = imresize(label, scalar_scale=1 / scale, method='bicubic')

            lr = np.moveaxis(lr, -1, 0)  # 텐서 계산을 위해 차원축 이동
            lr = np.expand_dims(lr, axis=0)  # 텐서 계산을 위해 차원 확장
            lr = torch.tensor(lr, dtype=torch.float32) / 255.
            label = label[border_cut:-border_cut, border_cut:-border_cut, :] / 255. # LR과 비교를 위해 크롭

            # 연산
            image_data = lr.to(device)  # LR
            outputs = model(image_data, scale, border_cut)

            save_image(outputs, f"test/testx3/{image}.png")

            outputs = outputs.cpu().detach().numpy()
            outputs = outputs.reshape(outputs.shape[2], outputs.shape[3], outputs.shape[1])

            sum_psnr += PSNR(label, outputs)  # 각 image의 pnsr 누적
            print(f"{image}: {PSNR(label, outputs):.3f}")

    print(f"average PSNR: {sum_psnr/len(images):.3f}")