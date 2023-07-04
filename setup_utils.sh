pip install -r GeoNeRF/requirements.txt
sudo apt update && sudo apt install tmux nvtop
# Install tmux-beautify
git clone https://github.com/gpakosz/.tmux.git ~/.oh-my-tmux \
	&& echo "set -g mouse on" >> ~/.oh-my-tmux/.tmux.conf \
	&& ln -s -f ~/.oh-my-tmux/.tmux.conf ~/.tmux.conf \
	&& cp ~/.oh-my-tmux/.tmux.conf.local ~/.tmux.conf.local