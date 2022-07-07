# Variational Autoencoder for Protein Structure Prediction

## Dihedral angles

<div align="center">
  <a href="https://docs.google.com/presentation/d/13_RwaMP5484OTUfon9xkMnEpmJlNzRAfPP2WxHcv2fE/edit#slide=id.p">
  <img width="620px" height="320px" src="https://user-images.githubusercontent.com/34074691/177210318-489ba43c-c2cf-4bd7-b557-413fdd928fd5.gif"></a>
</div>
<div align="center">
  <a href="https://proteopedia.org/wiki/index.php/Tutorial:Ramachandran_principle_and_phi_psi_angles">
  <img width="140px" height="35px" src="https://user-images.githubusercontent.com/34074691/177212096-201c20b3-528d-420b-a6f6-a2dd87dea908.png"></a>
</div>

<div align="center">
  <p font size =13px;">
    Dihedral Angles: <strong>Phi(φ&deg;)</strong> and <strong>Psi(ψ&deg;)</strong> in Alanine </p>
</div>

In a polypeptide chain, each amino acid has two main chain bonds that can rotate around alpha carbon (Ca) defining the Dihedral (or torsion) angles.
The simple principle that two atoms can not occupy the same space, defines a great deal of protein molecular structure.
In fact, most combinations of **φ&deg;** and **ψ&deg;** are impossible due to those steric effects.

## Variational Autoencoder architectures and training procedures

We aim to model amino acid dihedral angles (𝜑, 𝜓). More specifically, we touch an “artistic” model that has its roots in Bayesian inference and graphical models and provides an interesting framework for generating new data, similar to the dataset that it was trained on.
Deploying Pyro, a Deep Probabilistic Programming Language, we approach the problem with two Variational Autoencoder (VAE) architectures.

<div align="center">
  <a href="https://pyro.ai/examples/vae.html">
  <img width="520px" height="230px" src="https://user-images.githubusercontent.com/34074691/177275727-958533de-9039-44a4-9cd5-ac9f9fd5d736.png"></a>
</div>

This repository contains the code only for the Conditional Variational Autoencoder (CVAE), as it is a more advanced, more useful and probably more interesting version of the simple VAE.

Two different training procedures were followed trying to figure out if mixing deterministic with probabilistic training could end up in more rich results.
The mix training with annealed Kullback–Leibler divergence showed improved results.


## Evaluation with Ramachandran plots

<div align="center">
  <a href="https://en.wikipedia.org/wiki/Ramachandran_plot">
  <img width="400px" height="360px" src="https://user-images.githubusercontent.com/34074691/177722528-8aeb8f88-e612-42c2-ae06-38f585db6506.jpg"></a>
</div>

### Real angles
Real amino amino acid dihedral angles are plotted in Ramachandran plots for side by side comparison purposes with the generated angles.                                                                                                                             
<p float="left">
  <img src="https://user-images.githubusercontent.com/34074691/177670309-5eeb67e2-4652-45ee-a43a-d56c74b3e908.png" width="300" /> 
  <img src="https://user-images.githubusercontent.com/34074691/177670313-86d8ac39-8a8b-4422-aa0c-e1ea8a6f739b.png" width="300" />
  <img src="https://user-images.githubusercontent.com/34074691/177670312-0b06782c-3593-401d-8cfb-f6c3d4a501b0.png" width="300" />
</p>
                                                                                                                              
                                                                                                                              
## Results - Generated angles   
                                                                                                                              
<p float="left">
  <img src="https://user-images.githubusercontent.com/34074691/177724492-44caa23e-f35e-424a-95fb-2b2f9c72c6c6.png"  width="300" /> 
  <img src="https://user-images.githubusercontent.com/34074691/177718951-6ead9091-3532-47e3-a428-7088ff5791bb.png" width="300" />
  <img src="https://user-images.githubusercontent.com/34074691/177718953-d517f958-5496-46d7-a690-09f8e1e068c6.png" width="300" />
</p>                                                                                                                         




