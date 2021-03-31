0. ssh-copy-id

ssh-copy-id root@gpu1

https://askubuntu.com/questions/466549/bash-home-user-ssh-authorized-keys-no-such-file-or-directory

1. clone and checkout to branch

https://stackoverflow.com/questions/67699/how-to-clone-all-remote-branches-in-git

git pull --all
git branch -a
git checkout --track origin/sly
git status

2. cd supervisely/remote-dev
3. ./start.sh

4. ssh root@gpu1 -p 7777

to fix WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!
ssh-keygen -R [66.248.205.57]:7777