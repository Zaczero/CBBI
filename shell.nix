{ }:

let
  # Update with `nixpkgs-update` command
  pkgs =
    import
      (fetchTarball "https://github.com/NixOS/nixpkgs/archive/41965737c1797c1d83cfb0b644ed0840a6220bd1.tar.gz")
      { };

  pythonLibs = with pkgs; [
    stdenv.cc.cc.lib
    zlib.out
  ];
  python' =
    with pkgs;
    (symlinkJoin {
      name = "python";
      paths = [ python313 ];
      buildInputs = [ makeWrapper ];
      postBuild = ''
        wrapProgram "$out/bin/python3.13" --prefix LD_LIBRARY_PATH : "${lib.makeLibraryPath pythonLibs}"
      '';
    });

  packages' = with pkgs; [
    python'
    coreutils
    curl
    gnused
    jq
    ruff
    uv

    (writeShellScriptBin "nixpkgs-update" ''
      set -e
      hash=$(
        curl -fsSL \
          https://prometheus.nixos.org/api/v1/query \
          -d 'query=channel_revision{channel="nixpkgs-unstable"}' |
          jq -r ".data.result[0].metric.revision"
      )
      sed -i "s|nixpkgs/archive/[0-9a-f]\\{40\\}|nixpkgs/archive/$hash|" shell.nix
      echo "Nixpkgs updated to $hash"
    '')
    (writeShellScriptBin "docker-build-push" ''
      set -e
      if command -v podman &> /dev/null; then docker() { podman "$@"; } fi
      docker push $(docker load < $(nix-build --no-out-link) | sed -En 's/Loaded image: (\S+)/\1/p')
    '')
  ];

  shell' = ''
    export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=""
    export TZ=UTC

    if [ -f .env ]; then
      echo "Loading .env file"
      set -a; . .env; set +a
    else
      echo "Skipped loading .env file (not found)"
    fi

    current_python=$(readlink -e .venv/bin/python || echo "")
    current_python=''${current_python%/bin/*}
    [ "$current_python" != "${python'}" ] && rm -rf .venv/

    echo "Installing Python dependencies"
    export UV_PYTHON="${python'}/bin/python"
    uv sync --frozen
    source .venv/bin/activate
    export UV_PYTHON="$VIRTUAL_ENV/bin/python"
  '';
in
pkgs.mkShell {
  buildInputs = packages';
  shellHook = shell';
}
