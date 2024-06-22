#!/usr/bin/env bash

set -euo pipefail

_echo() {
    echo "$@" > /dev/tty
}


_echo
_echo "Checking identities..."

# read user input, assign stdin to keyboard
exec < /dev/tty

GIT_BRANCH=$(git branch --show-current)
remote_identities=$(git log origin/$GIT_BRANCH --pretty=format:"%an <%ae>" | sort | uniq)
_echo "Current Git Branch: $GIT_BRANCH"
_echo "Identities on remote:"
_echo "$remote_identities"
_echo

local_identities=$(git log $GIT_BRANCH --pretty=format:"%an <%ae>" | sort | uniq)
current_author="$(git config user.name) <$(git config user.email)>"
local_identities=$(echo -e "$local_identities\n$current_author" | sort | uniq)
_echo "Identities on local:"
_echo "$local_identities"
_echo

# Find all identities in local_identities that are not in remote_identities
new_identities=$(comm -23 <(echo "$local_identities") <(echo "$remote_identities"))

if [ -z "$new_identities" ]; then
    _echo "No new identities found."
    exit 0
else
    _echo "New identities found:"
    _echo "$new_identities"
    while true; do
        _echo -n "Did you intend to use this identity? (Y/N) "
        read yn < /dev/tty
        case $yn in
            [Yy] ) exit 0; break;;
            [Nn] ) _echo "Aborting"; exit 1;;
            * ) _echo "Please answer y (yes) or n (no):" && continue;;
        esac
    done
fi


exit 1

exec <&-
