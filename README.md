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

## Variational Autoencoder architectures and training precedures

We aim to model amino acid dihedral angles (𝜑, 𝜓). More specifically, we touch an “artistic” model that has its roots in Bayesian inference and graphical models and provides an interesting framework for generating new data, similar to the dataset that it was trained on.
Deploying Pyro, a Deep Probabilistic Programming Language, we approache the problem with two
Variational Autoencoder (VAE) architectures.

<div align="center">
  <a href="https://pyro.ai/examples/vae.html">
  <img width="520px" height="230px" src="https://user-images.githubusercontent.com/34074691/177275727-958533de-9039-44a4-9cd5-ac9f9fd5d736.png"></a>
</div>
