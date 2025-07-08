import os
import shutil
from pathlib import Path
import random
import matplotlib.pyplot as plt
import cv2


def prepare_dataset_folders(src_path, output_base="dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    categories=["Normal","Stone"]
    src_path=Path(src_path)

    for category in categories:
        all_files=list((src_path/category).glob("*.JPG"))
        print(f"{category} - {len(all_files)} files")
        random.shuffle(all_files)

        train_end=int(len(all_files)*train_ratio)
        val_end=int(len(all_files)*val_ratio)
        print(f"train_end: {train_end} - val_end : {val_end}")
        dataset_split = {    
            "train":all_files[:train_end],
            "validate":all_files[train_end:train_end+val_end],
            "test":all_files[train_end+val_end:]
        }

        print(f"train:{len(all_files[:train_end])} files")
        print(f"validate:{len(all_files[train_end:train_end+val_end])} files")
        print(f"test:{len(all_files[train_end+val_end:])} files")

        for split , files in dataset_split.items():
            dest_dir = Path(output_base)/split/category
            dest_dir.mkdir(parents=True,exist_ok=True)
            for file in files:
                shutil.copy(file,dest_dir/file.name)



def show_sample_images(folder_path,lable,n=4):
    path_dir=folder_path / lable
    print(f"folder_path : {path_dir } in show_sample_images")
    
    files = random.sample(os.listdir(path_dir),n)
    fig,axes=plt.subplots(1,n,figsize=(15,4))
    for i, file in enumerate(files):
        img_path= os.path.join(path_dir,file)
        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        axes[i].imshow(img,cmap='gray')
        axes[i].set_title(f"{lable}-{file}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()