{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell rec {
  buildInputs = with pkgs; [
    stdenv.cc.cc.lib
    gnumake
    gnused
    python311
    pipenv
  ];

  shellHook = with pkgs; ''
    export LD_LIBRARY_PATH="${lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
    export PIPENV_VENV_IN_PROJECT=1
    export PIPENV_VERBOSITY=-1
    [ -v DOCKER ] && [ ! -f ".venv/bin/activate" ] && pipenv sync
    [ ! -v DOCKER ] && [ ! -f ".venv/bin/activate" ] && pipenv sync --dev
    [ ! -v DOCKER ] && exec pipenv shell --fancy
  '';
}
