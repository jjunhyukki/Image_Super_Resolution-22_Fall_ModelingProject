import wandb
import torch
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR

save_dir = os.path.join(os.getcwd(), 'results')


def train(epochs,
          device,
          model,
          criterion,
          optimizer,
          scheduler,
          trainloader,
          validloader,
          log_iter,
          save_epoch,
          log_wandb,
          wandb_project=None,
          wandb_entity=None):

    if log_wandb:
        config = {
            'epochs': epochs,
            'batch_size': trainloader.batch_size,
            'learning_rate': optimizer.defaults['lr']
        }
        wandb.init(project=wandb_project, entity=wandb_entity, config=config)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.
        train_psnr = 0.
        print(f'Epoch {epoch + 1}')
        for idx, (input, target) in enumerate(tqdm(trainloader)):
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_psnr += PSNR(target.detach().cpu().numpy(),
                               output.detach().cpu().numpy())
            if (idx + 1) % log_iter == 1:
                print(
                    f'Loss After {idx + 1} iterations : {train_loss / (idx + 1)}')
                print(
                    f'PSNR After {idx + 1} iterations : {train_psnr / (idx + 1)}')
                print(f"Current LR : {optimizer.param_groups[0]['lr']}")
                wandb.log({'Train Loss': train_loss / (idx + 1),
                           'Train PSNR': train_psnr / (idx + 1),
                           'Learning Rate': optimizer.param_groups[0]['lr']})

        print(
            f'Loss After Epoch {epoch + 1} : {train_loss / len(trainloader)}')
        print(
            f'PSNR After Epoch {epoch + 1} : {train_psnr / len(trainloader)}')

        model.eval()
        valid_loss = 0.
        valid_psnr = 0.
        with torch.no_grad():
            for input, target in tqdm(validloader):
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = criterion(output, target)
                valid_loss += loss.item()
                valid_psnr += PSNR(target.detach().cpu().numpy(),
                                   output.detach().cpu().numpy())

        print(
            f'Valid Loss After Epoch {epoch + 1} : {valid_loss / len(validloader)}')
        print(
            f'Valid PSNR After Epoch {epoch + 1} : {valid_psnr / len(validloader)}')

        if log_wandb:
            wandb.log({'Train Loss per Epoch': train_loss / len(trainloader),
                       'Train PSNR per Epoch': train_psnr / len(trainloader),
                       'Valid Loss per Epoch': valid_loss / len(validloader),
                       'Valid PSNR per Epoch': valid_psnr / len(validloader)})

        if (epoch + 1) % save_epoch == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'CustomModel_{epoch+1}.pt'))
            torch.save({'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()},
                       os.path.join(save_dir, f'CustomModel_Checkpoint_{epoch+1}.tar'))
    if log_wandb:
        wandb.finish()
