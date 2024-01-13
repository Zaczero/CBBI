{ pkgs ? import <nixpkgs> { }, ... }:

with pkgs; let
  shell = import ./shell.nix {
    isDevelopment = false;
  };

  python-venv = pkgs.buildEnv {
    name = "python-venv";
    paths = [
      (pkgs.runCommand "python-venv" { } ''
        mkdir -p $out/lib
        cp -r "${./.venv/lib/python3.12/site-packages}"/* $out/lib
      '')
    ];
  };
in
with pkgs; dockerTools.buildLayeredImage {
  name = "zaczero/cbbi";
  tag = "latest";
  maxLayers = 10;

  contents = shell.buildInputs ++ [ python-venv ];

  extraCommands = ''
    set -e
    mkdir app && cd app
    cp "${./.}"/LICENSE .
    cp "${./.}"/*.py .
    mkdir api metrics
    cp "${./api}"/*.py api
    cp "${./metrics}"/*.py metrics
    export PATH="${lib.makeBinPath shell.buildInputs}:$PATH"
    ${shell.shellHook}
  '';

  config = {
    WorkingDir = "/app";
    Env = [
      "LD_LIBRARY_PATH=${lib.makeLibraryPath shell.buildInputs}"
      "PYTHONPATH=${python-venv}/lib"
      "PYTHONUNBUFFERED=1"
      "PYTHONDONTWRITEBYTECODE=1"
    ];
    Volumes = {
      "/app/output" = { };
    };
    Entrypoint = [ "python" "main.py" ];
    Cmd = [ ];
  };
}
