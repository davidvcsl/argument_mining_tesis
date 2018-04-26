#!/bin/bash
wget https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/data/argument-recognition/ArgumentAnnotatedEssays-1.0.zip
unzip ArgumentAnnotatedEssays-1.0.zip
cd ArgumentAnnotatedEssays-1.0
unzip brat-project.zip
cd ..
mv ArgumentAnnotatedEssays-1.0/brat-project example_essays