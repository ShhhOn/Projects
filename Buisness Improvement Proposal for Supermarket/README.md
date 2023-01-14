CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Description
 * Requirements
 * Usage

INTRODUCTION
------------

This package contains team017's Final Project for MGT6203 for the Spring 2022 semester.
The team members are: Ziyang Guo, Sean Lee, Vakula Mallapally, Bill Szeto, Tianhua Zhu

Our project focused on analyzing a set of publicly available marketing data in 
hopes of uncovering customer spending and purchasing habits and any correlation 
they may have with successive promotional campaigns and availability of various 
shopping methods.

Description
------------
In this package, you can find the following files:
* final report (Team 17 Final Report.docx) 
* the R Notebook (MGT6203Team17Project.Rmd)
* the knitted output (MGT6203Team17Project.html)
* the dataset (marketing_data.csv).


REQUIREMENTS
------------

To properly process and analyze the data, we require the following R packages, 
which are listed at the beginning of the attached R Notebook. Please make sure 
you have installed them before proceeding.
if (!require(dplyr)) install.packages("dplyr")
if (!require(corrplot)) install.packages("corrplot")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(GGally)) install.packages("GGally")
if (!require(ROCR)) install.packages("ROCR")
if (!require(car)) install.packages("car")
if (!require(matrixStats)) install.packages("matrixStats")
if (!require(mlbench)) install.packages("mlbench")
if (!require(caret)) install.packages("caret")
if (!require(clustMixType)) install.packages("clustMixType")
if (!require(pscl)) install.packages("pscl")

Usage
------------

Before running the code, please put the dataset (marketing_data.csv) in the same 
directory. The R code computationally performs the following tasks: 
* data loading and cleaning
* transform and make new variables
* descriptive analysis including correlation and outlier detection
* modeling (K-Prototypes, regression, classification)
Please cross-reference the final report, in which we described each task in more 
detail, adding interpretation and business implications. 



