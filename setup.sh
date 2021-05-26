#!/bin/bash

pip install -r requirements.txt
git clone --recursive https://github.com/getpelican/pelican-plugins
git clone https://github.com/VorpalBlade/pelican-cite.git

./build.sh
