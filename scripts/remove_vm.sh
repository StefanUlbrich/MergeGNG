#!/bin/sh

# https://stackoverflow.com/a/8426110
vm_dir=$(dirname "$(dirname "$(poetry run which python)")")

echo Removing $vm_dir


# https://stackoverflow.com/a/226724
while true; do
    read -p "You sure?" yn
    case $yn in
        [Yy]* ) rm -r $vm_dir; break;;
        [Nn]* ) echo "Exiting"; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done