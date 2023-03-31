# Are these descriptions referring to the same entity or just to similar ones?

## Introduction
This repository contains the code to reproduce the results of the our paper.
Due to the evaulation script being written in Java and our solution in Python it takes some extra steps to work.

##Setup
1. pip install -r pythonCompiler/oaei-resources/requirements.txt
1. compile the pythonCompiler project using maven (```mvn package```)
1. copy pythonCompiler/target/pythonCompiler-1.0-seals.zip to SealsMatcherRunner\src\main\resources\pythonCompiler-1.0-seals.zip
1. compile the SealsMatcherRunner project using maven (```mvn package```) and run the project (```mvn exec:java -D exec.mainClass=PythonEvaluator```)
1. output is in the results folder