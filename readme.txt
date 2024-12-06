Marcus Becker 
CSCI575 Final Project
Image Reconstruction from EEG Data

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

3. If starting from scratch, download images from the ImageNet Database. It may be necessary
to sign in with academic credentials in order to receive access. 
    imageNet_download.py
    TODO: In script
