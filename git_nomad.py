import os

import git
from pathlib import Path

remote_name = 'origin'
repo_path = Path('./nomad')

def pull_nomad(remote_path, branch_name='master'):
    try:
        repo = git.Repo(repo_path)
        repo_exists = True
        origin = repo.remotes.origin
    except git.NoSuchPathError:
        repo_path.mkdir()
        repo = git.Repo.init(repo_path)
        repo_exists = False
        origin = repo.create_remote(remote_name, remote_path)
        
    origin.fetch()

    if not repo_exists:
        repo.create_head(branch_name, origin.refs[branch_name])  # create local branch "master" from remote "master"
        repo.heads[branch_name].set_tracking_branch(origin.refs[branch_name])  # set local "master" to track remote "master
        repo.heads[branch_name].checkout()  # checkout local "master" to working tree

    origin.pull()
    
    return repo_path, repo

def get_local_time(utc_time):
    return
 