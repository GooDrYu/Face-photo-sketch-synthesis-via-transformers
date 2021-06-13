import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image,make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from datasets_unlined import *
from mixformer import *
import PIL.Image as Image
import lpips
import numpy as np

torch.cuda.set_device(0)
loss_fn = lpips.LPIPS(net='alex',version='0.1')
loss_fn.cuda()

transforms_ = [
    transforms.Resize([256,256], Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)


g=Generator().cuda()
g.load_state_dict(torch.load("/home/amax/mrzhu/wbyu/ctGAN_patchD/saved_models_0/G_75.pth"))
g.eval()
for param in g.parameters():
    param.requires_grad = False

img_dir =  "../{}/test/{}/".format('xm2vts','a')
img_filenames = [x for x in os.listdir(img_dir)]
comp_dir = "../{}/test/{}/".format('xm2vts','b')
gen_dir = "./img_test"

def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil = image_pil.resize((200, 250), Image.BICUBIC)
    image_pil.save(filename)
    #print("Image saved as {}".format(filename))
    
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
    


for img_name in img_filenames:
    img = load_img(img_dir + img_name)
    img = transform(img)
    input_img = img.unsqueeze(0). cuda()
    gen_img = g(input_img).detach().squeeze(0).cpu()
    if not os.path.exists("img_test"):
        os.mkdir("img_test")
    save_img(gen_img, "img_test/{}".format(img_name))


f = open('lpips.txt','w')
files = os.listdir(comp_dir)


for fil in files:
	if(os.path.exists(os.path.join(comp_dir,fil))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(comp_dir,fil))).cuda() # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(gen_dir,fil))).cuda()



		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		#print('%s: %.3f'%(fil,dist01))
		f.writelines('%s: %.6f\n'%(fil,dist01))

f.close()

score=np.loadtxt("./lpips.txt",delimiter=':',usecols=1).mean()
print(score)
