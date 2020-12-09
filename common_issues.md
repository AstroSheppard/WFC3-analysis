If using pyenv, make sure to install "brew install xz" before installing "pyenv install python version", otherwise there will be an issue with lmza missing and pandas (and other modules?) won't work.
After installing ipython within the virtual environment, make sure to either run "hash -r" or close shell and start a new one.
To make virtual environment: virtualenv venv. Switch into it via: source venv/bin/activate
pip freeze > requirements.txt ; pip install -r requirements.txt
matplotlib
