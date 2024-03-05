{ isDevelopment ? true }:

let
  # Currently using nixpkgs-23.11-darwin
  # Update with `nixpkgs-update` command
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/b3a5f534d8a260328c5e13bd81c19c0432afbe9f.tar.gz") { };

  libraries' = with pkgs; [
    # Base libraries
    stdenv.cc.cc.lib
    zlib.out
  ];

  # Wrap Python to override LD_LIBRARY_PATH
  wrappedPython = with pkgs; (symlinkJoin {
    name = "python";
    paths = [ python312 ];
    buildInputs = [ makeWrapper ];
    postBuild = ''
      wrapProgram "$out/bin/python3.12" --prefix LD_LIBRARY_PATH : "${lib.makeLibraryPath libraries'}"
    '';
  });

  packages' = with pkgs; [
    # Base packages
    wrappedPython
  ] ++ lib.optionals isDevelopment [
    # Development packages
    poetry
    ruff

    # Scripts
    # -- Misc
    (writeShellScriptBin "nixpkgs-update" ''
      set -e
      hash=$(git ls-remote https://github.com/NixOS/nixpkgs nixpkgs-23.11-darwin | cut -f 1)
      sed -i -E "s|/nixpkgs/archive/[0-9a-f]{40}\.tar\.gz|/nixpkgs/archive/$hash.tar.gz|" shell.nix
      echo "Nixpkgs updated to $hash"
    '')
    (writeShellScriptBin "docker-build-push" ''
      set -e
      if command -v podman &> /dev/null; then docker() { podman "$@"; } fi
      docker push $(docker load < $(nix-build --no-out-link) | sed -En 's/Loaded image: (\S+)/\1/p')
    '')
  ];

  shell' = with pkgs; lib.optionalString isDevelopment ''
    [ ! -e .venv/bin/python ] && [ -h .venv/bin/python ] && rm -r .venv

    echo "Installing Python dependencies"
    export POETRY_VIRTUALENVS_IN_PROJECT=1
    poetry env use "${wrappedPython}/bin/python"
    poetry install --no-root --compile

    echo "Activating Python virtual environment"
    source .venv/bin/activate

    # Development environment variables
    export PYTHONNOUSERSITE=1

    if [ -f .env ]; then
      echo "Loading .env file"
      set -o allexport
      source .env set
      +o allexport
    fi
  '';
in
pkgs.mkShell {
  buildInputs = libraries' ++ packages';
  shellHook = shell';
}
