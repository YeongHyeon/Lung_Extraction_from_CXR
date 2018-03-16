<h1>Lung Extraction from CXR</h1>

This repository contains 'Lung Extaction System'. It need for efficient computing in registration between previous CXR and post CXR or etc. It is great for extracting 'Lung area' while minimizing loss of information in the original and cutting off unnecessary information. Its performance measured as following.</br>
<strong>Precision</strong>: 0.788</br>
<strong>Recall</strong>: 1.0</br>
<strong>F1-score</strong>: 0.881</br>  

<h2>Flow Chart</h2>
<p align="center">
  <img src="./readme/flowchart.png" alt="Flowchart" id="flow" title="Flowchart"><br>
  Fig 1. The flowchart of the system.
</p>

<h2>CNN Structure</h2>
<p align="center">
  <img src="./readme/model.png" alt="ResNet" id="resnet" title="ResNet" style="width: 400px;"><br>
  Fig 2. The CNN model for classify each regions. It constructed ResNet like model.
</p>

<h2>Sample of Dataset</h2>
<p align="center">
  <img src="./readme/datasample.png" alt="Datasample" id="datasample" title="Datasample"><br>
  Fig 3. Sample data of training dataset. These are reflects shape and texture of data that pre-processed by contouring and masking.
</p>
Dataset: <a hfef="https://nihcc.app.box.com/v/ChestXray-NIHCC">Chest X-ray8</a> from NIH. (Total 112120 CXR images.)

<h2>Result</h2>
<p align="center">
  <img src="./readme/result.png" alt="Result" id="result" title="result"><br>
  Fig 4. Result of the system. The blue box is ROI of CXR that interested on lung. The red vox is ROI of diagnosis that annotated by NIH.
</p>
