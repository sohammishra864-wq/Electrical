# Project Prompt: RNN-Based Electrical Load Classification Using RL, RC, and RLC Circuit Datasets

I want to develop a complete Electrical Engineering project using Artificial Intelligence (AI), specifically an **RNN (Recurrent Neural Network) classifier**, for electrical load classification using simulated electrical circuit data.

## 1. Dataset Generation

Generate and simulate datasets using the following circuit configurations:

- RL Circuit
- RC Circuit
- RLC Circuit

Collect time-series electrical data from the simulations, including:

- Voltage
- Current
- Active Power
- Reactive Power (where applicable)
- Apparent Power
- Impedance
- Phase Angle
- Frequency

Generate sufficient samples under different operating conditions to make the dataset suitable for training, validation and testing of the RNN classifier.

## 2. Load Classification

Classify the load characteristics into the following three categories:

- CP – Constant Power
- CC – Constant Current
- CV – Constant Voltage

Use time-series voltage and current signals along with other relevant electrical parameters as input to the AI model.

## 3. RNN Classifier Development

Implement an **RNN (Recurrent Neural Network) classifier** for electrical load classification.

- Design and implement an RNN-based classification model.
- Use time-series voltage and current signals as input to the RNN.
- Develop the appropriate input layer, recurrent hidden layers and output layer.
- Train the RNN to classify CP, CC and CV loads.
- Use appropriate activation functions, loss functions and optimization algorithms.
- Split the dataset into training, validation and testing datasets.
- Apply preprocessing and normalization where required.
- Prevent data leakage during training and testing.
- Evaluate the model using unseen test data.
- Aim for at least 95% classification accuracy and less than 5% classification error, if achievable.

## 4. Feature Extraction and Data Preprocessing

Extract relevant electrical features from the simulated signals, such as:

- RMS Voltage
- RMS Current
- Active Power
- Reactive Power
- Apparent Power
- Power Factor
- Voltage and Current variation over time
- Frequency-related features

Explain the preprocessing and feature extraction process in simple language.

## 5. AI Model Performance Evaluation

Evaluate the RNN classifier using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Error Percentage

Generate graphs showing the model's performance and classification results.

## 6. Graphical Analysis

Generate graphs for:

- Voltage vs Time
- Current vs Time
- Power vs Time
- Voltage and Current Waveforms
- Feature Analysis Graphs
- Actual vs Predicted Classification
- Confusion Matrix
- RNN Model Accuracy
- RNN Training and Validation Loss
- Classification Performance Comparison, if applicable

## 7. Excel Output

Export the complete dataset, classification results, performance metrics and graphs into an Excel (.xlsx) workbook.

Create separate Excel sheets for:

1. Complete Dataset
2. RL Dataset
3. RC Dataset
4. RLC Dataset
5. Feature Extraction
6. Actual vs Predicted Results
7. RNN Performance Metrics
8. Confusion Matrix
9. Graphical Analysis

Include properly formatted tables, graphs, column headings, units and legends.

## 8. Project Architecture

Develop a complete project architecture and block diagram illustrating the following workflow:

Circuit Simulation (RL, RC, RLC) → Dataset Generation → Data Preprocessing → Feature Extraction → RNN Classifier → CP/CC/CV Classification → Performance Evaluation → Excel Tables and Graphs.

## 9. Final Deliverables

Provide the following:

- Complete project methodology.
- RL, RC and RLC circuit simulation models.
- Dataset generation procedure.
- Data preprocessing and feature extraction.
- RNN classifier architecture.
- Complete RNN implementation code.
- Training and testing results.
- Accuracy, precision, recall and F1-score.
- Confusion matrix.
- Graphical analysis.
- Excel workbook containing all datasets, results, tables and graphs.
- Project architecture/block diagram.
- Step-by-step explanation of the entire project in simple Electrical Engineering language.

Use MATLAB/Simulink or Python, whichever is most suitable for implementing the RNN classifier and generating the Excel output. Explain the entire implementation step by step.
