Marcus Becker 
CWID: 10873051
CSCI575 Final Project
Image Reconstruction from EEG Data

Using the python programming language, various dependencies

All source files are located in the MB_EEG_ML/src/ directory.
Code is executed by running various scripts in the MB_EEG_ML/scripts/ directory
-init_all.sh- Script to start from scratch. GRADER SHOULD NOT NEED TO RUN


This project was developed on an M1 Pro Mac. While everything should be platform agnostic, the code is configured for this
architecture, and a different setup may require troubleshooting.

How to Run:

1. Ensure Virtual Environment is setup with Python 3.12
    python3.12 -m venv venv
    source ./venv/bin/activate

2. Ensure requirements libraries are all installed in the virtual environment 
    May need to install torch FIRST, ELSE YOU MAY HAVE ENVIRONMENT ISSUES

    pip install torch torchvision torchaudio
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
    pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
    pip install torch-geometric

    pip install -r requirements.txt

3. IF JUST RUNNING CODE, SKIP TO STEP 4, IF STARTING FROM SCRATCH (NOT RECOMMENDED):
    i. Download the actual EEG-ImageNet Dataset file from https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/
        This is an 8GB file, so i cannot provide it. It should be saved at MB_EEG_ML/scripts/data/EEG-ImageNet.pth

    ii. Run the MB_EEG_ML/scripts/init_all.sh script (while inside the MB_EEG_ML/scripts/ directory). This will do a few things
        -Downloads image groups from the ImageNet Database using the imageNet_download.py script. It may be necessary
        to sign in with academic credentials at https://image-net.org/download-images.php in order to receive access. 
        This is necessary because of different copyright
        and usage rules for the EEG-ImageNet and ImageNet Datasets. 

        -Generates a list of images, then rescales and prunes the dataset for missing images, using the gen_img_list.py script 
        and tools from process_images.py. This scales all images to 224x224, and handles any potential missing images from the
        EEG-ImageNet dataset by pruning and re-saving it. 

        -Runs the BLIP-CLIP encoding script to generate caption descriptions for each image.
        Outputs the clip_embeddings.pth file, and captions.txt file.
        You may need to connect to a Huggingface account in order to use the remote models





    
    
