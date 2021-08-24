#!/bin/bash

pytest -v -s
echo "Printing contents of local dir"
ls -la
echo "Printing contents of media/"
ls -la media/
cat media/report_mobilenetv2.md
