{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    python310Packages.jupyterlab
    python310Packages.pytorch
    python310Packages.torchvision
    python310Packages.tkinter
    python310Packages.matplotlib
    python310Packages.pyqt6-sip
    python310Packages.pygobject3
    gobject-introspection
    gtk3 
  ];
}
