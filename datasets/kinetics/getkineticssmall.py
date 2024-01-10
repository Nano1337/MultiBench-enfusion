"""Code to eventually load kinetics data."""
import os
import shutil
from pathlib import Path
import torch

from torchvision.datasets import Kinetics400


def getkinetics(datafolder, tempfolder, categorylist, frames_per_instance, reallabel, frame_skip=1, centercrop=None):
    """ UNUSED: TODO

    Args:
        datafolder (_type_): _description_
        tempfolder (_type_): _description_
        categorylist (_type_): _description_
        frames_per_instance (_type_): _description_
        reallabel (_type_): _description_
        frame_skip (int, optional): _description_. Defaults to 1.
        centercrop (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # TODO
    # for category in categorylist:
    #     os.system("mv "+datafolder+"/"+category+" "+tempfolder)
    a = Kinetics400(tempfolder, 300, extensions=('mp4',))
    datas = []
    print("Total videos: "+str(len(a)))
    for ii in range(len(a)):
        (video, audio, label) = a[ii]
        vh = len(video[0])
        vw = len(video[0][0])
        v = video.view(-1, frame_skip, vh, vw, 3)[:, 0, :, :, :].squeeze()
        if centercrop is not None:
            w, h = centercrop
            if(w > vw) or (h > vh):
                continue
            hstart = (vh-h)//2
            hend = hstart+h
            wstart = (vw-w)//2
            wend = wstart+w
            v = v[:, hstart:hend, wstart:wend, :]
        alen = len(audio[0])  # TODO this is wrong, should be 1
        print(len(v))
        '''
        ap=alen*frames_per_instance*frame_skip//300
        for i in range(len(v)//frames_per_instance):
            vi=v[i*frames_per_instance:(i+1)*frames_per_instance]
            alen=len(audio[0])
            ai=audio[ap*i:ap*(i+1),:]
            print(vi.shape)
            print(ai.shape)
            datas.append((vi,ai,reallabel))
        '''
    exit()

    for category in categorylist:
        os.system("mv "+tempfolder+"/"+category+" "+datafolder)
    return datas



def get_data(categories, base_dir, splitsize=50):
    """
    Process and organize files in given categories.

    Args:
        categories (list): List of category names.
        base_dir (str): Base directory of data.
        splitsize (int): Number of files to process in a batch. Defaults to 50.
    """

    for category in categories:
        category_dir = Path(base_dir) / category
        temp_dir = Path(base_dir) / 'temp'

        if not category_dir.exists():
            print(f"Category directory {category_dir} not found.")
            continue

        temp_dir.mkdir(parents=True, exist_ok=True)

        files = list(category_dir.glob('*'))
        for i in range(0, len(files), splitsize):
            batch_files = files[i:i + splitsize]
            for file in batch_files:
                shutil.copy(file, temp_dir)

            # Process the batch (Replace with actual processing code)
            # process_batch(batch_files, temp_dir)

            for file in batch_files:
                shutil.move(temp_dir / file.name, category_dir)

        # Additional processing if needed
        # post_process_category(category_dir)

# Example usage
get_data(['archery', 'breakdancing', 'crying', 'dining', 'singing'], '/home/pliang/yiwei/kinetics/ActivityNet/Crawler/Kinetics/test_data')
