"""
Script used to convert original cell crops present in: https://virtualcellmodels.cziscience.com/dataset/hpa-subcellular-section-subcell with reduced resolution into a smaller dataset
"""

HPA_data_path = "/scr/data/cell_crops"
save_path = "/scr/data/mini-hpa"

import os
from skimage import io, transform
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

png_paths = (
    os.popen(f"find {HPA_data_path} -name '*cell_image.png'").read().strip().split("\n")
)

print("Number of PNG images", len(png_paths))

for png_path in tqdm(png_paths):
    img = io.imread(png_path)  # skimage.io.imread supports 4 channels
    img_resized = transform.resize(
        img, (256, 256), preserve_range=True, anti_aliasing=True
    ).astype(img.dtype)
    sub_folder = os.path.dirname(png_path).replace(HPA_data_path, save_path)
    os.makedirs(sub_folder, exist_ok=True)
    io.imsave(os.path.join(sub_folder, os.path.basename(png_path)), img_resized)

    def process_image(png_path):
        img = io.imread(png_path)
        img_resized = transform.resize(
            img, (256, 256), preserve_range=True, anti_aliasing=True
        ).astype(img.dtype)
        sub_folder = os.path.dirname(png_path).replace(HPA_data_path, save_path)
        os.makedirs(sub_folder, exist_ok=True)
        io.imsave(os.path.join(sub_folder, os.path.basename(png_path)), img_resized)

    print("CPU count:", cpu_count())
    if __name__ == "__main__":
        num_cores = 94
        with Pool(num_cores) as pool:
            list(
                tqdm(
                    pool.imap_unordered(process_image, png_paths), total=len(png_paths)
                )
            )
