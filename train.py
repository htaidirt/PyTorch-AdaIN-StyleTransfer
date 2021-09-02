import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from PIL import Image
from tqdm.notebook import tqdm
import numpy as np
import os
import sys
from Utils import networks

manualSeed = 999
torch.manual_seed(manualSeed)

path = "/home/ec2-user/app"
sys.path.append(path)


class Images(Dataset):
  """
  Images Dataset that returns one style and one content image. As I only trained using 40.000
  images each, each image is randomly sampled. The way it is implemented does not allow multi-threading. However
  as this network is relatively small and training times low, no improved class was implemented.
  """
  def __init__(self, root_style_images, root_content_images, transform=None):
    self.root_style_images = root_style_images
    self.root_content_images = root_content_images
    self.transform = transform

  def __len__(self):
    return min(len(os.listdir(self.root_style_images)), len(os.listdir(self.root_content_images)))

  def __getitem__(self, idx):
    all_names1, all_names2 = os.listdir(self.root_style_images), os.listdir(self.root_content_images)
    idx1, idx2 = np.random.randint(0, len(all_names1)), np.random.randint(0, len(all_names2))

    img_name1, img_name2 = os.path.join(self.root_style_images, all_names1[idx1]), os.path.join(self.root_content_images, all_names2[idx2])
    image1 = Image.open(img_name1).convert("RGB")
    image2 = Image.open(img_name2).convert("RGB")

    if self.transform:
      image1 = self.transform(image1)
      image2 = self.transform(image2)

    return image1, image2


# To note is that the images are not normalised
transform = transforms.Compose([transforms.Resize(512),
                               transforms.CenterCrop(256),
                               transforms.ToTensor()])

# Specify the path to the style and content images
pathStyleImages = "data/wikiart"
pathContentImages = "data/coco_images"

all_img = Images(pathStyleImages, pathContentImages, transform=transform)

# Simple save
def save_state(decoder, optimiser, iters, run_dir):
  name = "models/StyleTransfer_Checkpoint_Iter_{}.tar".format(iters)
  torch.save({"Decoder" : decoder,
              "Optimiser" : optimiser,
              "iters": iters
              }, os.path.join(path, name))
  print("Saved : {} succesfully".format(name))


def training_loop(network, # StyleTransferNetwork
                  dataloader_comb, # DataLoader
                  n_epochs, # Number of Epochs
                  run_dir # Directory in which the checkpoints and tensorboard files are saved
                  ):
  writer = SummaryWriter(os.path.join(path, run_dir))
  # Fixed images to compare over time
  fixed_batch_style, fixed_batch_content = all_img[0]
  fixed_batch_style, fixed_batch_content =  fixed_batch_style.unsqueeze(0).to(device), fixed_batch_content.unsqueeze(0).to(device) # Move images to device

  writer.add_image("Style", torchvision.utils.make_grid(fixed_batch_style))
  writer.add_image("Content", torchvision.utils.make_grid(fixed_batch_content))

  iters = network.iters

  for epoch in range(1, n_epochs+1):
    tqdm_object = tqdm(dataloader_comb, total=len(dataloader_comb))

    for style_imgs, content_imgs in tqdm_object:
      network.adjust_learning_rate(network.optimiser, iters)
      style_imgs = style_imgs.to(device)
      content_imgs = content_imgs.to(device)

      loss_comb, content_loss, style_loss = network(style_imgs, content_imgs)

      network.optimiser.zero_grad()
      loss_comb.backward()
      network.optimiser.step()

      # Update status bar, add Loss, add Images
      tqdm_object.set_postfix_str("Combined Loss: {:.3f}, Style Loss: {:.3f}, Content Loss: {:.3f}".format(
                                  loss_comb.item()*100, style_loss.item()*100, content_loss.item()*100))

      if iters % 25 == 0:
        writer.add_scalar("Combined Loss", loss_comb*1000, iters)
        writer.add_scalar("Style Loss", style_loss*1000, iters)
        writer.add_scalar("Content Loss", content_loss*1000, iters)

      # if (iters+1) % 50 == 1:
      #   with torch.no_grad():
      #     network.set_train(False)
      #     images = network(fixed_batch_style, fixed_batch_content)
      #     img_grid = torchvision.utils.make_grid(images)
      #     writer.add_image("Progress Iter: {}".format(iters), img_grid)
      #     network.set_train(True)

      if (iters+1) % 100 == 1:
        print(f"Now on iters={iters}")

      if (iters+1) % 500 == 1:
          save_state(network.decoder.state_dict(), network.optimiser.state_dict(), iters, run_dir)
          writer.close()
          writer = SummaryWriter(os.path.join(path, run_dir))

      iters += 1

device = ("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4
learning_rate_decay = 5e-5

dataloader_comb = DataLoader(all_img, batch_size=5, shuffle=True, num_workers=0, drop_last=True)
gamma = torch.tensor([2.0]).to(device) # Style weight. Gamma = 2

n_epochs = 5
run_dir = "runs/Run_1" # Change if you want to save the checkpoints/tensorboard files in a different directory

state_encoder = torch.load(os.path.join(path, "resources/vgg_normalised.pth"))
network = networks.StyleTransferNetwork(device,
                                        state_encoder,
                                        learning_rate,
                                        learning_rate_decay,
                                        gamma,
                                        load_fromstate=False,
                                        load_path=os.path.join(path, "models/StyleTransfer_Checkpoint_Iter_40000.tar"),
                                       )

training_loop(network, dataloader_comb, n_epochs, run_dir)
