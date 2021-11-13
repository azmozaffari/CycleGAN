#CycleGan
import torch
import torch.nn as nn
import torchvision.transforms as transforms 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from Util import Generator 
from Util import Discriminator
from Util import ImageLoader
############################################################         Data Preperation   ###################################
# read data and make the datasets


# define transformer

transform ={"train": transforms.Compose([transforms.Resize((270)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
	"test":transforms.Compose([transforms.Resize((256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}



# download the dataset

# train_dataset = datasets.ImageFolder('./data/horse2zebra/train', transform["train"])
train_dataset = ImageLoader('./data/horse2zebra/train/horse/','./data/horse2zebra/train/zebra/',transform["train"])

# define  the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)




###########################################################   Define the model   ########################################

# criterion

epochs = 200

G_horse = Generator().to('cuda')
G_zebra = Generator().to('cuda')
D_horse = Discriminator().to('cuda')
D_zebra = Discriminator().to('cuda')

lr = 0.0002


param = list(G_horse.parameters())+list(G_zebra.parameters()) 
optim_G = torch.optim.Adam(param,lr, betas=(0.5, 0.999))
optim_D_h = torch.optim.Adam(D_horse.parameters(), lr, betas=(0.5, 0.999))
optim_D_z = torch.optim.Adam(D_zebra.parameters(), lr, betas=(0.5, 0.999))

###########################
scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=100, gamma=0.1)
scheduler_D_h = torch.optim.lr_scheduler.StepLR(optim_D_h, step_size=100, gamma=0.1)
scheduler_D_z = torch.optim.lr_scheduler.StepLR(optim_D_z, step_size=100, gamma=0.1)

##############################

####################Adam vs sgd#####################




L1_loss = torch.nn.L1Loss().to('cuda')
MSE_loss = torch.nn.MSELoss().to('cuda')





for e in range(epochs):
	loss_dis_zebra = 0
	loss_dis_horse = 0
	loss_gen = 0
	for data in train_dataloader:
		

		images_horse = data["horse"].to('cuda')
		images_zebra = data["zebra"].to('cuda')


		labels_horse = torch.ones((len(images_horse),1)).to('cuda')
		labels_zebra = torch.ones((len(images_zebra),1)).to('cuda')
		
		fake_labels_horse = torch.ones((len(labels_horse),1)).to('cuda')
		fake_labels_zebra = torch.ones((len(labels_zebra),1)).to('cuda')



		# optimize Generators
		optim_G.zero_grad()

		


		img_fake_zebra = G_horse(images_horse)
		fake_zebra_label = D_zebra(img_fake_zebra)

		img_fake_horse = G_zebra(images_zebra)
		fake_horse_label = D_horse(img_fake_horse)

		fake_fake_horse = G_zebra(img_fake_zebra)
		fake_fake_zebra = G_horse(img_fake_horse)

		loss_G_zebra = L1_loss(fake_fake_zebra,images_zebra)
		loss_G_horse = L1_loss(fake_fake_horse,images_horse)

		loss = 10*(loss_G_zebra+loss_G_horse)+L1_loss(D_horse(img_fake_horse),fake_labels_horse)+L1_loss(D_zebra(img_fake_zebra),fake_labels_zebra)+MSE_loss(G_zebra(images_horse),images_horse)+ MSE_loss(G_horse(images_zebra),images_zebra)
		loss_gen += loss.data
		loss.backward()
		optim_G.step()

	
	# for data in train_dataloader:
	# 	images_horse = data["horse"].to('cuda')
	# 	images_zebra = data["zebra"].to('cuda')


	# 	labels_horse = torch.ones((len(images_horse),1)).to('cuda')
	# 	labels_zebra = torch.ones((len(images_zebra),1)).to('cuda')
		
	# 	fake_labels_horse = torch.ones((len(labels_horse),1)).to('cuda')
	# 	fake_labels_zebra = torch.ones((len(labels_zebra),1)).to('cuda')



		# optimize discrimator horse
		img_fake_horse = G_zebra(images_zebra)
		optim_D_h.zero_grad()
		labels_horse_1 = torch.zeros((len(images_horse),1)).to('cuda')
		
		img = torch.cat((images_horse,img_fake_horse),0)
		lab = torch.cat((labels_horse,labels_horse_1),0)
		loss1 = MSE_loss(D_horse(img),lab)
		loss1.backward()
		optim_D_h.step()
		loss_dis_horse += loss1.data
		
		# optimize discrimator zebra
		img_fake_zebra = G_horse(images_horse)
		labels_zebra_1 = torch.zeros((len(images_zebra),1)).to('cuda')
		
		img = torch.cat((images_zebra,img_fake_zebra),0)
		lab = torch.cat((labels_zebra,labels_zebra_1),0)
		optim_D_z.zero_grad()
		loss2 = MSE_loss(D_zebra(img),lab)
		loss2.backward(retain_graph=True)
		optim_D_z.step()
		loss_dis_zebra += loss2.data





	scheduler_G.step()
	scheduler_D_h.step()
	scheduler_D_z.step()

	print("epoch = ", e)
	print("loss_gen = ",loss_gen)
	print("loss_dis_zebra = ", loss_dis_zebra)
	print("loss_dis_horse = ", loss_dis_horse)


torch.save(G_zebra.state_dict(), "./saved_model/G_zebra.pth")
torch.save(G_horse.state_dict(), "./saved_model/G_horse.pth")




