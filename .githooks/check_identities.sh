#!/usr/bin/env bash

# exec < /dev/tty

# Check for a current list of identities using git shortlog -sne --all, ignoring any staged commits
GIT_BRANCH=$(git branch --show-current)
remote_identities=$(git log origin/$GIT_BRANCH --pretty=format:"%an <%ae>" | sort | uniq)
echo "Current Git Branch: $GIT_BRANCH"
echo "Identities on remote:"
echo "$remote_identities"
echo
# local_identities=$(git log $GIT_BRANCH --pretty=format:"%an <%ae>" | sort | uniq)
current_author="$(git config user.name) <$(git config user.email)>"
local_identities=$(echo -e "$local_identities$current_author" | sort | uniq)
echo "Identities on local:"
echo "$local_identities"
echo

# Find all identities in local_identities that are not in remote_identities
new_identities=$(comm -23 <(echo "$local_identities") <(echo "$remote_identities"))

if [ -n "$new_identities" ]; then
    echo "You are introducing the new identities:"
    echo "$new_identities"
    echo
    while true; do
        # prompt for user input
        read -p "Is this intentional? (Y/n)" yn < /dev/tty
        if [ "$yn" = "" ]; then
            yn='n'
        fi
        case $yn in
          [Y] ) echo "Identity checks passed"; exit 0;;
          [Nn] ) echo "Aborting due to identity leak";exit 1;;
          * ) echo "Please answer Y for yes or n for no.";;
        esac
    done
else
    echo "No new identities found"
    exit 0
fi
