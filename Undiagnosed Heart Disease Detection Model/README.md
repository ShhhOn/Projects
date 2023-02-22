PROJECT TITLE
-------------
Using Patient Medical Journey Data to Identify Undiagnosed Heart Disease

DESCRIPTION
-----------
This package uses anonymized Medicare Claims Data to analyze geographical locations and trends that may help doctors and researchers identify
patients with undiagnosed heart disease. This package first harvests Medicare Claims SynPUF data for 2008-2010 in the OMOP Common Data Model
from OHDSI with 2.3 million patients aged 65 and older using a Databricks Spark Cluster. 

Next, analytical models are built using both R and python (scikit learn package). In R, we test a robust Logistic Regression approach with 
data exploration, variable selection, cost analysis, and goodness of fit testing. In python, we test Logistic Regression side-by-side with
approaches including: Random Forests, Decision Trees, KNN, Gaussian Naive Bayes, and SVM. Results from the R and python experiments are
combined for a single, streamlined feed into Tableau for visualization.

Finally, model results are loaded into Tableau for visualization by State and geographic region, with the ability to drill down and experiment
with results across various demographics and risk factors used in the predictive models. The purpose is to identify patients who may have
heart disease but are not yet diagnosed. Undiagnosed heart disease is defined as false positives from our model: The model thinks the patient
has heart disease, but the actual data shows the patient is not diagnosed with heart disease.

INSTALLATION
------------
1. Create (or login to) Databricks account for Execution steps 1-4
2. Download Python and install Jupyter Notebook and Pandas package for Execution step 5
3. Download and install R and R studio for Execution steps 6 and 9
4. Set up Python Jupyter Notebook For Execution steps 5, 7 and 8
5. Download Tableau (license required) for Execution step 10.

EXECUTION
---------
1.  Download raw data files for the 2.3 million dataset. Download URLs are available in 0.raw_file_download_urs.txt
2.  Convert the compression of *.lzo files to *.gz and upload to Databricks environment. Databricks does not support LZO compression by default
3.  Using Databricks environment, load respective subject areas into their own tables using PySpark jobs in 
    1.load_raw_data_and_generate_curated_dataset/load_claims_data_*.ipynb. HTML versions of these files available in the same location for review.
4.  Using Databricks environment, run the data curation logic in PySpark notebook 1.load_raw_data_and_generate_curated_dataset/processing.dbc 
    to generate a curated dataset of 2.3 million patients (heart_disease_conditions_23M_20220319.csv.gz). processing.html available in the same 
    location for review
5.  Using Jupyter Notebook and Pandas, run 1.load_raw_data_and_generate_curated_dataset/split_23M_file_into_100K_22M.ipynb to split the 2.3 million
    patients curated dataset created in the earlier step into two datasets, a 100K dataset (heart_disease_conditions_100K_20220407.csv) and a 2.2 million
    dataset(heart_disease_conditions_22M_20220407.csv). split_23M_file_into_100K_22M.html available in the same location for review
6.  Using R, run Logistic_Regression_R_Final.Rmd which loads in the input generated in Step 5 and produces analysis results for Logistic Regression
    Output = data_export_hda.RData
7.  Using Python Jupyter Notebook, run file "Classification Models_100K train_2.2M test_Final.ipynb" which loads in the input generated in Step 5 
    and produces analysis results for Logistic Regression, Random Forests, Decision Tree, KNN, Gaussian Naive Bayes, and SVM models. 
    Output = "heart_disease_conditions_model_predictions_2.2M.csv"
8.  Using Python Jupyter Notebook, run file "Classification Models_Decision Boundaries.ipynb" which loads the output 
    "heart_disease_conditions_22M_20220407.csv" from step 5, performs Principal Component Analysis on the dataset to identify the top 2 principal
    components, and generates decision boundary graphs for the classification models. 
9.  Using R, run Merge_Results.Rmd which combines the outputs of Steps 6 and 7 as inputs and merges the data to produce csv files ready
    to be consumed by Tableau. Output = hda.csv and hdas.csv
10. Using Tableau, open the dashboard titled "Undiagnosed Heart Disease in Medicare Patients 2008-2010.twbx". Alternatively, navigate to the following
    URL to interact with the dashboard:
    https://public.tableau.com/app/profile/kipp.spanbauer/viz/UndiagnosedHeartDiseaseinMedicarePatients2008-2010/UndiagnosedHeartDiseaseinMedicarePatients