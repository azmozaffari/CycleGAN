import os
from PIL import Image
from torch.utils.data import Dataset


class ImageLoader(Dataset):
    
    def __init__(self,input_path_horse,input_path_zebra,transform_image=None):
        
        self.input_path_horse = input_path_horse
        self.input_path_zebra = input_path_zebra
        self.transform_image = transform_image
        
        
    def getImagesNames(self,path):
      
        
        images_names = os.listdir(path)        
        return images_names
    
    def __len__(self):
        return max(len( os.listdir(self.input_path_zebra)),len( os.listdir(self.input_path_zebra)))
    
    def __getitem__(self, idx):
        file_names_zebra = self.getImagesNames(self.input_path_zebra)
        file_names_horse = self.getImagesNames(self.input_path_horse)
        



        file_path_z = self.input_path_zebra+file_names_zebra[idx%len( os.listdir(self.input_path_zebra))]
        file_path_h = self.input_path_horse+file_names_horse[idx%len( os.listdir(self.input_path_horse))]

        img_z = self.transform_image(Image.open(file_path_z).convert('RGB'))
        img_h = self.transform_image(Image.open(file_path_h).convert('RGB'))
        
        
        return {"horse":img_h,"zebra":img_z}