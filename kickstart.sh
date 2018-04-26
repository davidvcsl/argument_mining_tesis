#!/bin/bash
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
mkdir tmp
mkdir pkl