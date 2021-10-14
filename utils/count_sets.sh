#!/bin/bash
cd ../dataset/AITEX
train=$(ls ./trainset | wc -l)
test=$(ls ./testset | wc -l)
validation=$(ls ./validationset | wc -l)
echo Trainset: $train
echo Validationset: $validation
echo Testset: $test