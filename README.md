Lung Extraction from CXR
========================

This repository contains 'Lung Extaction System'.</br>
It need for efficient computing in registration between previous CXR and post CXR or etc. It is great for extracting 'Lung area' while minimizing loss of information in the original and cutting off unnecessary information.</br>
Its performance measured as following.</br>
<strong>Precision</strong>: 0.788</br>
<strong>Recall</strong>: 1.0</br>
<strong>F1-score</strong>: 0.881</br>  

Flow Chart
----------
<figure>
  <center>
    <img src="./readme/flowchart.png" alt="Flowchart" id="flow" title="Flowchart" style="width: 300px;">
    <figcaption>Fig 1. The flowchart of the system.</figcaption>
  </center>
</figure>  

CNN Structure
-------------
<figure>
  <center>
    <img src="./readme/model.png" alt="ResNet" id="resnet" title="ResNet" style="width: 400px;">
    <figcaption>Fig 2. The CNN model for classify each regions. It constructed ResNet like model.</figcaption>
  </center>
</figure>  

Sample of Dataset
-----------------

<figure>
  <center>
    <img src="./readme/datasample.png" alt="Datasample" id="datasample" title="Datasample">
    <figcaption>Fig 3. Sample data of training dataset. These are reflects shape and texture of data that pre-processed by contouring and masking.</figcaption>
  </center>
</figure>  

Dataset: <a hfef="https://nihcc.app.box.com/v/ChestXray-NIHCC">Chest X-ray8</a> from NIH. (Total 112120 CXR images.)

Result
------
<figure>
  <center>
    <img src="./readme/result.png" alt="Result" id="result" title="result">
    <figcaption>Fig 4. Result of the system. The blue box is ROI of CXR that interested on lung. The red vox is ROI of diagnosis that annotated by NIH.</figcaption>
  </center>
</figure>
