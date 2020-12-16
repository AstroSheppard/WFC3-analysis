If using pyenv, make sure to install "brew install xz" before installing "pyenv install python version", otherwise there will be an issue with lmza missing and pandas (and other modules?) won't work.
After installing ipython within the virtual environment, make sure to either run "hash -r" or close shell and start a new one.
To make virtual environment: virtualenv venv. Switch into it via: source venv/bin/activate
pip freeze > requirements.txt ; pip install -r requirements.txt
matplotlib, numpy, scipy, astropy, and a few others no longer supported old version downloads and had to be downloaded with python 3. 
ipython still referencing system ipython within virtualenv

To switch branches: git checkout NEW_BRANCH
To confirm: git branch

To merge specific files between branches:
1) Switch to branch you want files to be merged into
2) Fetch/merge branch to make sure git up to date (git pull)
3) git checkout OTHER_BRANCH PATH_TO_FILES
4) commit changes
5) push back to branch you merged into

Example: Create data directory and merge csv files from python2 branch into main (python3) branch

git checkout main

git pull origin main

git checkout python2 data/l9859b/*.csv

git commit -m "Merge some files"

git push origin main
