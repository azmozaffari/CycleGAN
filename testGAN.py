import torch
import torch.nn as nn
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from Util import Generator 
from Util import ImageLoader
from torchvision.utils import save_image


# define transformer

transform ={"train": transforms.Compose([transforms.Resize((256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
	"test":transforms.Compose([transforms.Resize((256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}



# download the dataset

# train_dataset = datasets.ImageFolder('./data/horse2zebra/train', transform["train"])
train_dataset = ImageLoader('./data/horse2zebra/test/horse/','./data/horse2zebra/test/zebra/',transform["train"])

# define  the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)


G_horse = Generator().to('cuda')

G_zebra = Generator().to('cuda')


G_horse.load_state_dict(torch.load("./saved_model/G_horse.pth"))
G_horse.eval()


G_zebra.load_state_dict(torch.load("./saved_model/G_zebra.pth"))
G_zebra.eval()



id = 0


for i,data in enumerate(train_dataloader):
		

	images_horse = data["horse"].to('cuda')
	images_zebra = data["zebra"].to('cuda')

	fake_zebra = 0.5*(G_horse(images_horse)+1)
	fake_horse = 0.5*(G_zebra(images_zebra)+1)

	save_image(images_horse[id], 'img_real_horse'+str(i)+'.png')
	save_image(fake_horse[id], 'img_fake_horse'+str(i)+'.png')
	save_image(images_zebra[id], 'img_real_zebra'+str(i)+'.png')
	save_image(fake_zebra[id], 'img_fake_zebra'+str(i)+'.png')


	




