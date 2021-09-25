#!/bin/bash

# nice function
exit_check() {
    if [ $? -eq 0 ] 
    then 
        echo "SUCCEES"
    else 
        echo "FAILURE"
        exit 255
    fi
}


echo "Running cmd tests..."

echo "Testing mobilenetv2"
train --model_name="mobilenetv2" --parameters_path="test_parameters.yml" --debug="False"
exit_check

echo "Testing xception"
train --model_name="xception" --parameters_path="test_parameters.yml" --debug="False"
exit_check

echo "Testing resnext101_32x4d"
train --model_name="resnext101_32x4d" --parameters_path="test_parameters.yml" --debug="False"
exit_check

echo "Testing vggm"
train --model_name="vggm" --parameters_path="test_parameters.yml" --debug="False"
exit_check

echo "Testing inceptionv4"
train --model_name="inceptionv4" --parameters_path="test_parameters.yml" --debug="False"
exit_check
