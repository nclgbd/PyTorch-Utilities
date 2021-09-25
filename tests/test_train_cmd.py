import os
import sys
import subprocess

def run_mobilenetv2_cmd():
    p = subprocess.Popen(f'train --model_name="mobilenetv2" --parameters_path="test_parameters.json" --debug="False"', shell=True)
    return p.returncode

def test_run_mobilenetv2_cmd():
    assert run_mobilenetv2_cmd() == 0