# Confidence-based-Prediction-of-Antibiotic-Resistance

Rapid and accurate diagnostics of bacterial infections are necessary for efficient treatment of antibiotic-resistant pathogens. Cultivation-based methods, such as antibiotic susceptibility testing (AST), are slow, resource-demanding, and can fail to produce results before the treatment needs to start. This increases patient risks and antibiotic overprescription. Here, we present a deep-learning method that uses transformers to merge patient data with available AST results to predict antibiotic susceptibilities that have not been measured. The method is combined with conformal prediction (CP) to enable the estimation of uncertainty at the patient-level. After training on three million AST results from thirty European countries, the method made accurate predictions for most antibiotics while controlling the error rates, even when limited diagnostic information was available. We conclude that transformers and CP enables confidence-based decision support for bacterial infections and, thereby, offer new means to meet the growing burden of antibiotic resistance.


## Software dependencies
- Python 3.9.5
- PyTorch 1.11.0
- CUDA 11.3.1
- IPython 7.25.0 
- GCCcore 10.3.0
- PyYAML 5.4.1 

## GPU
The model was ran on a GPU server using one NVIDIA Tesla A100 HGX GPU. 

## Running
- Save the folder [torch_app](https://github.com/indajuan/Confidence-based-Prediction-of-Antibiotic-Resistance/tree/main/code/torch_app) and the [vocabulary file](https://github.com/indajuan/Confidence-based-Prediction-of-Antibiotic-Resistance/blob/main/code/vocabulary_obj.pth) in the working directory.
- The [example dataset](https://github.com/indajuan/Confidence-based-Prediction-of-Antibiotic-Resistance/edit/main/code/example_input_file.csv) contains the basic structure of an input file.
   - Replace Age, Country, and Gender valid values present in the [vocabulary file](https://github.com/indajuan/Confidence-based-Prediction-of-Antibiotic-Resistance/blob/main/code/vocabulary_obj.pth)
- An example of the model's architecture, data points creation, training and testing parameters can be found in the [model's configuration file](https://github.com/indajuan/Confidence-based-Prediction-of-Antibiotic-Resistance/blob/main/code/model2/config.yaml)

- The command 
```
python train.py <model_folder> config.yaml
```
will run the script train.py, taking the configuration file *config.yaml* present in the model's folder *<model_folder>*, build and train the model with the parameters and datasets specified there. After training, the model will be saved in the model's folder. The output of the command are five files containing the training loss (pretrain_train_loss.pt), training accuracy (pretrain_train_acc.pt), training state (pretrain_state.pkl), and two models, the transformer model (pretrain_transformer.torch) and the classification neural networks (pretrain_classification.torch). 

- The command 
```
python final_val.py <model_folder> config.yaml
```
will run the script final_val.py, taking the training, testing and validation data sets to create data points and do predictions of antibiotic susceptibility tests results on them. The output of the command are three files train_final.csv, test_final.csv, and validation_final.csv, containing the predictions on the data points created from the corresponding data sets in the format:
[Country, Age, Gender, Date, and at least 4 antibiotic susceptibility tests results separated by commas]", <antibiotic predicted>, 1,0,"[-0.1634320616722107, -0.7023367285728455]"


