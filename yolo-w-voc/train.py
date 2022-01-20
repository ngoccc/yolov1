import torch
import torch.optim as optim
import time
from datetime import datetime
from pathlib import Path

from config import args
from yolo import MyYOLO
from dataloader import get_dataloader
from tools import gt_creator

def train():
    # Basic settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    root = '.'
    now = datetime.now()
    result_dir = Path(root) / 'results' / args.name
    ckpt_dir = result_dir / args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = result_dir / args.log_dir / now.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard.
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    epoch = 0
    global_step = 0
    best_accuracy = 0.

    # Network
    net = MyYOLO(device, input_size=args.train_size, num_classes=args.num_classes, trainable=True)
    net.to(device)
    # print(net)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume the training 
    if args.resume:
        ckpt_path = ckpt_dir / ('%s.pt' % args.ckpt_reload)

        try:
            checkpoint = torch.load(ckpt_path)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint['best_accuracy']
            print(f'>> Resume training from epoch {epoch+1}')

        except Exception as e:
            print(e)

    # Get train/test data loaders  
    dataloader = get_dataloader(args)
    # print(len(dataset))

    # Start training
    # Save the starting time
    # start_time = time.time()

    for epoch in range(epoch, args.epoch):
        # start time
        _start_time = time.time() 

        # Here starts the train loop.
        net.train()
        for batch_idx, (imgs, targets) in enumerate(dataloader):

            global_step += 1

            # Send imgs, targets to either cpu or gpu using `device` variable. 
            imgs = imgs.to(device)
            
            # Make train labels
            targets = [label.tolist() for label in targets]
            targets = gt_creator(input_size=args.train_size, stride=net.stride, label_lists=targets)
            targets = torch.tensor(targets).float().to(device)
            print(batch_idx)

            # Forward and loss
            optimizer.zero_grad()
            conf_loss, cls_loss, txtytwth_loss, total_loss = net(imgs, target=targets)
            
            # Backprop
            total_loss.backward()
            
            optimizer.step()

            if global_step % args.log_iter == 0 and writer is not None:
                # Log `loss` with a tag name 'train_loss' using `writer`. Use `global_step` as a timestamp for the log. 
                writer.add_scalar('object loss', conf_loss.item(), global_step)
                writer.add_scalar('class loss', cls_loss.item(), global_step)
                writer.add_scalar('local loss', txtytwth_loss.item(), global_step)
            
            continue
        '''
        t = time.time()-_start_time
        print(f'Epoch {epoch}/{args.epoch} [ Loss: obj {conf_loss.item():.4f} || cls {conf_loss.item():.4f}'
                f'bbox {txtytwth_loss.item():.4f} || total {total_loss.item():.4f} ] time={t:.3f} secs')
        '''

if __name__ == '__main__':
    train()