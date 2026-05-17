import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pycocotools.coco import COCO
from model.PAD_unet import UNet
import torch
import matplotlib.patches as mpatches
from utils.PAD_datamodule import PADDataModule
from utils.settings.config import CROP_ENCODING, IMG_SIZE, LINEAR_ENCODER, BANDS

parser = argparse.ArgumentParser()
parser.add_argument('--load_checkpoint', type=str, required=True)
parser.add_argument('--root_path_coco', type=str, default='coco_files/')
parser.add_argument('--medians_path', type=str, required=True)
parser.add_argument('--netcdf_path', type=str, default='dataset/netcdf')
parser.add_argument('--img_size', nargs='+', required=True)
parser.add_argument('--bands', nargs='+', default=sorted(list(BANDS.keys())))
parser.add_argument('--window_len', type=int, default=6)
parser.add_argument('--fixed_window', action='store_true', default=False)
parser.add_argument('--requires_norm', action='store_true', default=False)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_idx', nargs='+', required=False)
args = parser.parse_args()

args.img_size = tuple(map(int, args.img_size))
root_path_coco = Path(args.root_path_coco)
path_test = root_path_coco / 'coco_test.json'
run_path = Path(*Path(args.load_checkpoint).parts[:-2])
checkpoint_epoch = Path(args.load_checkpoint).stem.split('=')[1].split('-')[0]
print(f'Exportando a: {run_path}')

model = UNet.load_from_checkpoint(
    args.load_checkpoint,
    map_location=torch.device('cpu'),
    run_path=run_path,
    linear_encoder=LINEAR_ENCODER,
    checkpoint_epoch=checkpoint_epoch,
    num_layers=3
)

dm = PADDataModule(
    netcdf_path=Path(args.netcdf_path),
    path_test=path_test,
    group_freq='1MS',
    prefix=None,
    bands=args.bands,
    linear_encoder=LINEAR_ENCODER,
    saved_medians=True,
    medians_path=Path(args.medians_path),
    window_len=args.window_len,
    fixed_window=args.fixed_window,
    requires_norm=args.requires_norm,
    return_masks=False,
    clouds=False, cirrus=False, shadow=False, snow=False,
    output_size=args.img_size,
    batch_size=1,
    num_workers=args.num_workers,
    binary_labels=False,
    return_parcels=False
)
dm.setup('test')
model.cuda()
model.eval()

total_images = len(COCO(path_test).imgs)
image_idx = [int(x) for x in args.image_idx] if args.image_idx else [np.random.randint(0, total_images)]

num_subpatches = (IMG_SIZE // args.img_size[0], IMG_SIZE // args.img_size[1])

for image_id in image_idx:
    fig, axes = plt.subplots(1, 2, figsize=(30, 15))
    subpatch_id = image_id * (num_subpatches[0] * num_subpatches[1])

    grid1 = ImageGrid(fig, 121, nrows_ncols=num_subpatches, axes_pad=0.0)
    grid2 = ImageGrid(fig, 122, nrows_ncols=num_subpatches, axes_pad=0.0)

    for idx in range(num_subpatches[0] * num_subpatches[1]):
        batch = dm.dataset_test.__getitem__(subpatch_id + idx)
        im = grid1[idx].imshow(batch['labels'].squeeze(), vmin=0, vmax=max(LINEAR_ENCODER.values()), cmap='tab20')
        grid1[idx].set_axis_off()

        inputs = batch['medians'][None, :, :, :, :]
        inputs = torch.from_numpy(inputs).cuda()
        b, t, c, h, w = inputs.size()
        inputs = inputs.view(b, -1, h, w)
        with torch.no_grad():
            pred = model(inputs).to(torch.float32)

        pred_sparse = pred.argmax(axis=1).squeeze().cpu().numpy()
        grid2[idx].imshow(pred_sparse, vmin=0, vmax=max(LINEAR_ENCODER.values()), cmap='tab20')
        grid2[idx].set_axis_off()

    crop_encoding_rev = {v: k for k, v in CROP_ENCODING.items()}
    crop_encoding = {k: crop_encoding_rev[k] for k in LINEAR_ENCODER.keys() if k != 0}
    crop_encoding[0] = 'Background/Other'
    crop_ids = sorted(LINEAR_ENCODER.keys())
    colors = [im.cmap(im.norm(LINEAR_ENCODER[c])) for c in crop_ids]
    patches = [mpatches.Patch(color=colors[LINEAR_ENCODER[c]], label=f'{c} ({crop_encoding[c]})') for c in crop_ids]
    axes[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-large')

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

    axes[0].set_title('Label', fontsize=22)
    axes[1].set_title('Prediction', fontsize=22)

    out = run_path / f'evaluation_image{image_id}_epoch{checkpoint_epoch}.png'
    plt.savefig(out, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
    print(f'Guardado: {out}')
