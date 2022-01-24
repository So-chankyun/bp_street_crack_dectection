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

from pan.pan_dataloading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from pan.networks import PAN, ResNet50, Mask_Classifier
import pan.ss_transforms as tr

dir_img = Path('./data/train/')
dir_mask = Path('./data/train_masks/')
dir_checkpoint = Path('./checkpoints/')

def train_net(net,
              res,
              mc,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.2,
              save_checkpoint: bool = True,
              img_scale: float = 0.2,
              transform=None,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale,transform=transform)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale,transform=transform)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    # 짝수일 경우
    if n_val % 2 == 0:
        n_val -= 1
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    model_name = ['res', 'pan', 'mc']
    optimizer = {'res': optim.SGD(res.parameters(), lr=args.lr, weight_decay=1e-4),
                 'net': optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-4),
                 'mc': optim.SGD(mc.parameters(), lr=args.lr, weight_decay=1e-4)}

    # goal: maximize Dice score
    optimizer_lr_scheduler = {'res': optim.lr_scheduler.ReduceLROnPlateau(optimizer['res'], 'max', patience=2),
                              'net': optim.lr_scheduler.ReduceLROnPlateau(optimizer['net'], 'max', patience=2),
                              'mc': optim.lr_scheduler.ReduceLROnPlateau(optimizer['mc'], 'max', patience=2)}

    # 기존의 optimizer 및 scheduler
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        res.train()
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                # assert images.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    fms_blob, z = res(images)
                    out_ss = net(fms_blob[::-1])
                    mask_pred = mc(out_ss)

                    loss = criterion(mask_pred, true_masks) \
                           + dice_loss(F.softmax(mask_pred, dim=1).float(),
                                       F.one_hot(true_masks, 2).permute(0, 3, 1, 2).float())
                                        # 차원을 섞어줌

                # Update model
                model_name = [res, net, mc]
                for m in model_name:
                    m.zero_grad()

                grad_scaler.scale(loss).backward()

                model_name = ['res', 'pan', 'mc']
                for m in model_name:
                    grad_scaler.step(optimizer[m])

                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            # histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            # histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, res, mc, val_loader, device,)

                        # 각 model 들의 learning rate를 val_score를 통해 조절
                        for m in model_name:
                            optimizer_lr_scheduler[m].step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        # experiment.log({
                        #     'learning rate': optimizer.param_groups[0]['lr'],
                        #     'validation Dice': val_score,
                        #     # 'images': wandb.Image(images[0].cpu()),
                        #     # 'masks': {
                        #     #     'true': wandb.Image(true_masks[0].float().cpu()),
                        #     #     'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                        #     # },
                        #     'step': global_step,
                        #     'epoch': epoch,
                        #     **histograms
                        # })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    train_transforms = transforms.Compose([tr.RescaleSized(512),
                                           tr.MinMax(255.0),
                                           tr.ToTensor()
                                           ])

    test_transforms = transforms.Compose([tr.RescaleSized(256),
                                               tr.MinMax(255.0),
                                               tr.ToTensor()
                                               ])

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    # 2. PAN
    convnet = ResNet50(pretrained=True)
    net = PAN(convnet.blocks[::-1])


    # logging.info(f'Network:\n'
    #              f'\t{net.n_channels} input channels\n'
    #              f'\t{net.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    NUM_CLASS = 1
    mask_classifier = Mask_Classifier(in_features=256, num_class=(NUM_CLASS + 1))

    convnet.to(device=device)
    net.to(device=device)
    mask_classifier.to(device=device)

    try:
        train_net(net=net,
                  res=convnet,
                  mc = mask_classifier,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  transform=train_transforms,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
