from skimage.metrics import structural_similarity as ssim


def calculate_ssim(img1, img2, tensors=True):
    if tensors:
        img1 = img1.permute(1,2,0).cpu().detach().numpy()
        img2 = img2.permute(1,2,0).cpu().detach().numpy()
    return  ssim(img1, img2, multichannel=True, data_range=img2.max() - img2.min())