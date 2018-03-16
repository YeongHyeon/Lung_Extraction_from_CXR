<h1>Lung Extraction from CXR</h1>

<p align="center">
  <b>Some Links:</b><br>
  <a href="#">Link 1</a> |
  <a href="#">Link 2</a> |
  <a href="#">Link 3</a>
  <br><br>
  <img src="http://s.4cdn.org/image/title/105.gif">
</p>

This repository contains 'Lung Extaction System'. It need for efficient computing in registration between previous CXR and post CXR or etc. It is great for extracting 'Lung area' while minimizing loss of information in the original and cutting off unnecessary information. Its performance measured as following.</br>
<strong>Precision</strong>: 0.788</br>
<strong>Recall</strong>: 1.0</br>
<strong>F1-score</strong>: 0.881</br>  

<h2>Flow Chart</h2>
<figure>
  <center>
    <img src="./readme/flowchart.png" alt="Flowchart" id="flow" title="Flowchart" style="width: 300px;">
    <figcaption>Fig 1. The flowchart of the system.</figcaption>
  </center>
</figure>  

<h2>CNN Structure</h2>
<figure>
  <center>
    <img src="./readme/model.png" alt="ResNet" id="resnet" title="ResNet" style="width: 400px;">
    <figcaption>Fig 2. The CNN model for classify each regions. It constructed ResNet like model.</figcaption>
  </center>
</figure>

<h2>Sample of Dataset</h2>
<figure>
  <center>
    <img src="./readme/datasample.png" alt="Datasample" id="datasample" title="Datasample">
    <figcaption>Fig 3. Sample data of training dataset. These are reflects shape and texture of data that pre-processed by contouring and masking.</figcaption>
  </center>
</figure>  


Dataset: <a hfef="https://nihcc.app.box.com/v/ChestXray-NIHCC">Chest X-ray8</a> from NIH. (Total 112120 CXR images.)

<h2>Result</h2>
<figure>
  <center>
    <img src="./readme/result.png" alt="Result" id="result" title="result">
    <figcaption>Fig 4. Result of the system. The blue box is ROI of CXR that interested on lung. The red vox is ROI of diagnosis that annotated by NIH.</figcaption>
  </center>
</figure>
