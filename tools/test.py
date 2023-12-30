import torch
import pandas as pd

from dataloaders import create_dataset, create_dataloader

def test(model, args):
    test_dataset = create_dataset(args.test_dirs, is_train=False)
    test_loader = create_dataloader(
        test_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, is_train=False
    )

    print("Start testing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    preds = []
    img_names = []
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            preds.append(model(images).cpu())
            img_names.extend(names)

    preds = torch.cat(preds, dim=0).numpy() # (B, C)

    # Save to dataframe
    df = pd.DataFrame(preds, columns=['HG', 'HT', 'TR', 'CTH', 'BD', 'VH', 'CTQ', 'DQT', 'KS', 'CVN'])
    df['name'] = img_names
    df = df[['name', 'HG', 'HT', 'TR', 'CTH', 'BD', 'VH', 'CTQ', 'DQT', 'KS', 'CVN']]
    if args.train_full:
        df.to_csv(f"prediction/{args.model}_full.csv", index=False)
    else:
        df.to_csv(f"prediction/{args.model}_val.csv", index=False)