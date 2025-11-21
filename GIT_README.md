# push_to_git

A small helper script that stages, commits, and pushes your work while **excluding** all `.csv`, `.pkl`, `.key`, and `.pdf` files, and always ignoring `census_data1.pkl`.

## Install

1. Save the script to `~/bin`:
   ```zsh
   mkdir -p ~/bin
   mv push_to_git ~/bin/
   chmod +x ~/bin/push_to_git

2. Add ~/bin to your PATH (if not already):

echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

push_to_git "commit message"
Copy code
