{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    python311Packages.jupyterlab
    python311Packages.pytorch
    python311Packages.pyrsistent
    python311Packages.mypy
    python311Packages.pydantic
    python311Packages.torchvision
    python311Packages.tkinter
    python311Packages.matplotlib
    python311Packages.pyqt6-sip
    python311Packages.pygobject3
    python311Packages.tqdm
    python311Packages.transformers
    python311Packages.datasets
    gobject-introspection
    gtk3 
  ];
}
