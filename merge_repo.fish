#!/usr/bin/env fish

# Helper script to merge other repository into the main repo while preserving history.

# Define the new repository name and the array of existing repository names
set old_repo "new_repo_name"
set old_repo_branch "main"
set old_repo_path "path_to_old_repository"
set new_repo_subdir "./subdirectory"

# Add the repository as a remote
git remote add $old_repo $old_repo_path

# Fetch the data from the repository
git fetch $old_repo

# Create a branch and checkout
set branch_name (string replace -r '/' '-' $old_repo)
git checkout -b $branch_name $old_repo/$old_repo_branch

# Move all files and folders into a new subdirectory
mkdir -p $new_repo_subdir 
for file in (git ls-tree --name-only $branch_name)
    git mv $file $new_repo_subdir 
end

git commit -m "Moved $old_repo files into subdir"

git checkout main
git merge $branch_name --allow-unrelated-histories

# cleaning up the local branch
git branch -d $branch_name
