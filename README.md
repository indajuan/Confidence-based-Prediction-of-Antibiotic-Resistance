# Confidence-based-Prediction-of-Antibiotic-Resistance

We present a deep-learning method that uses transformers to merge patient data with available AST results to predict antibiotic susceptibilities that have not been measured. After training on information from 1,161,303 _E. coli_, 301,506 _K. pneumoniae_, and 166,448 _P. aeruginosa_ isolates collected from blood infections of patients in 30 European countries, the method made accurate predictions for most antibiotics while controlling the error rates, even when limited diagnostic information was available. 


## Software dependencies
- Python 3.9.5
- PyTorch 1.11.0
- CUDA 11.3.1
- IPython 7.25.0 
- GCCcore 10.3.0
- PyYAML 5.4.1 

## Running
- Save the folder [torch_app](https://github.com/indajuan/Confidence-based-Prediction-of-Antibiotic-Resistance/tree/main/torch_app) and the [vocabulary file](https://github.com/indajuan/Confidence-based-Prediction-of-Antibiotic-Resistance/blob/main/data/vocabulary_obj_test2024.pth) in the data directory.

### Input file 
Input file using patient information should follow the format:
Strain, Patient_information, AST used as features, AST to predict. For example, 

```
ESCCOL, HR F 70 2014_10, CRO_S AMX_R CIP_S GEN_S, CAZ_S AMP_R
PSEAER, FR M 80 2013_10, IPM_S MEM_S TZP_S AMK_S PIP_S FEP_S CIP_S, GEN_S TOB_S CAZ_S
ESCCOL, UK F 66 2010_10, GEN_R CAZ_R CIP_R TOB_R, CTX_R
```

Input file to run the model without patient information should follow the format:
Strain, Patient_information, AST used as features, AST to predict. For example, 

```
ESCCOL, NA, CRO_S AMX_R CIP_S GEN_S, CAZ_S AMP_R
PSEAER, NA, IPM_S MEM_S TZP_S AMK_S PIP_S FEP_S CIP_S, GEN_S TOB_S CAZ_S
```

- The [example dataset with patient data](https://github.com/indajuan/Confidence-based-Prediction-of-Antibiotic-Resistance/edit/main/data/example_with_patient.csv) contains the basic structure of an input file with patient data.
- The [example dataset without patient data](https://github.com/indajuan/Confidence-based-Prediction-of-Antibiotic-Resistance/edit/main/data/example_without_patient.csv) contains the basic structure of an input file with patient data.
- The valid values for Age, Country, Gender, and Date of isolate sampleing are listed below, non valid words will be converted to a word representing an unkown word. If one variable is missing, it will instead be padded. It is recommended to leave the space empty. 
- The main configuration of the file should be placed inside each model
- Countries correspond to abbreviations of members of the European Centre for Disease Prevention and Control:
AT, BE, BG, CY, CZ, DE, DK, EE, EL, ES, FI, FR, HR, HU, IE, IS, IT, LT, LU, LV, MT, NL, NO, PL,  PT, RO, SE, SI, SK, UK
- Age: 0 to 120
- Gender: F, M
- Date: from 2007_01 (where 2007 is the year and 01 is the month), to 2020_9

- To run the model with patient data
```
python run_model.py model_with_patient_data config.yaml both data/input_with_patient_data.csv output_with_patient_data.csv
```

- To run the model without patient data
```
python run_model.py model_without_patient_data config.yaml antibiotic data/input_ithout_patient_data.csv output_without_patient_data.csv
```

This will run the script run_model.py, taking the configuration file *config.yaml* present in the model's folder *<model_folder>*, using either only the *antibiotic* or *both* (antibiotic's and patient's) transformers. The output of the command has the columns: id (initial row number), Antibiotic , AST_true (the known AST for the antibiotic), AST_prediction (AST prediction for the antibiotc), Antibiotic_predictors (AST fed to the model as input), Patient_data (if no patient data was used, this will return <unk>), Output_neural_networks (Scores to classify the isolate as susceptible or resistant to the antibiotic).

For example,
- With patient data
```
id, Antibiotic, AST_true, AST_prediction, Antibiotic_predictors, Patient_data, Output_neural_networks
1, AMC, R, R, ESCCOL CAZ_S AMP_R CIP_S CTX_S, IT F 88 2017_3, "[-0.1192, 0.092]"
```

- Without patient data
```
id,Y,Y_resp,Y_pred,Antibiotic_predictors,Patient_data,Output_neural_networks
1, AMC, R, R, ESCCOL CAZ_S AMP_R CIP_S CTX_S, <unk>, "[-0.1192, 0.092]"
```



