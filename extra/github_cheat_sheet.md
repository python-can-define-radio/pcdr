## Just setup
git remote add upstream https://...

## Run these as needed while main is checked out
git fetch upstream
git pull upstream main
git push

## Then to update the other
git checkout whatever-other-branch
git rebase main
git push
