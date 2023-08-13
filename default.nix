{ pkgs ? import <nixpkgs> { } }:

with pkgs; let
  shell = import ./shell.nix {
    inherit pkgs;
    isDocker = true;
  };

  python-venv = buildEnv {
    name = "python-venv";
    paths = [
      (runCommand "python-venv" { } ''
        mkdir -p $out/lib
        cp -r "${./.venv/lib/python3.11/site-packages}"/* $out/lib
      '')
    ];
  };
in
dockerTools.buildLayeredImage {
  name = "zaczero/cbbi";
  tag = "latest";
  maxLayers = 10;

  contents = shell.buildInputs ++ [ python-venv ];

  extraCommands = ''
    mkdir app && cd app
    cp "${./.}"/LICENSE .
    cp "${./.}"/*.py .
    mkdir api metrics
    cp "${./api}"/*.py api
    cp "${./metrics}"/*.py metrics
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
