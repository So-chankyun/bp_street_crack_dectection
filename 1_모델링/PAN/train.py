import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from utils.dataloading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from pan.networks import PAN
from unet import UNet
import utils.ss_transforms as tr

DATAPATH = "D:/crack data/도로장애물·표면 인지 영상(수도권)/Training/!CHANGE/CRACK/!changes/"
dir_img = Path(DATAPATH.replace("!CHANGE", "Images").replace("!changes","data"))
dir_mask = Path(DATAPATH.replace("!CHANGE", "Annotations").replace("!changes","data"))
dir_checkpoint = Path('./checkpoints/')

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 2,
              learning_rate: float = 0.0001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              amp: bool = False):

    transform = transforms.Compose([tr.RescaleSized((640, 384)),
                                           tr.MinMax(255.0),
                                           tr.ToTensor()
                                           ])

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, data_type='train', transform=transform)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, data_type='train', transform=transform)
=======
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

dir_img = Path('./data/train/')
dir_mask = Path('./data/train_masks/')
dir_checkpoint = Path('./checkpoints/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):\
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
<<<<<<< HEAD

    if n_train % 2 != 0:
        n_train += 1
        n_val = len(dataset) - n_train

=======
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
<<<<<<< HEAD
    train_loader = DataLoader(train_set, shuffle=False, **loader_args)
=======
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
<<<<<<< HEAD
                                  val_percent=val_percent, save_checkpoint=save_checkpoint,
=======
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
<<<<<<< HEAD
=======
        Images scaling:  {img_scale}
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
<<<<<<< HEAD

    # goal: maximize Dice score
    optimizer_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

=======
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
<<<<<<< HEAD

=======
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

<<<<<<< HEAD
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    mask_pred = net(images)

                    loss = criterion(mask_pred, true_masks.long().squeeze(1)) \
                           + dice_loss(F.softmax(mask_pred, dim=1).float(), F.one_hot(true_masks.long().squeeze(1), 2).permute(0, 3, 1, 2).float())
                    #                     # 차원을 섞어줌

                # Update model
                net.zero_grad()
=======
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad()
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
<<<<<<< HEAD
                    'epoch': epoch,
=======
                    'epoch': epoch
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
<<<<<<< HEAD
                division_step = (n_train // (2 * batch_size))
                export_img_step = (n_train // (10 * batch_size))

                if division_step > 0 or export_img_step > 0:
                    # 10% 마다 이미지 저장.
                    if global_step % export_img_step == 0:
                        experiment.log({
                             'images': wandb.Image(images[0].cpu()),
                             'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(mask_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                             }
                        })

                    # 1 epoch마다 validation 실시
=======
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
<<<<<<< HEAD

                        optimizer_lr_scheduler.step(val_score)
=======
                        scheduler.step(val_score)
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
<<<<<<< HEAD
=======
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
<<<<<<< HEAD
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            script_model = torch.jit.script(net)
            script_model.save(str(dir_checkpoint / 'checkpoint_epoch{}_torchscript.pt'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')

=======
            # 이 부분에 torchscript로 모델을 변경하는 코드를 삽입하자.(Annotation 방식이 될 것같다.)
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
<<<<<<< HEAD
=======
    parser.add_argument('--scale', '-s', type=float, default=0.2, help='Downscaling factor of the images')
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

<<<<<<< HEAD
    # 2. PAN
    NUM_CLASS = 1

    net = PAN(backbone='resnet34', pretrained=True, n_class=2)
    # net = UNet(n_channels=3,n_classes=2,bilinear=True)
=======
    net = UNet(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
<<<<<<< HEAD

=======
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
<<<<<<< HEAD
                  val_percent=args.val / 100,
                  amp=args.amp)

=======
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
>>>>>>> b21dbb31461ff9a9454c5184dcdb918fd063a3c7
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
