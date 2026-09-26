# To learn more about how to use Nix to configure your environment
# see: https://firebase.google.com/docs/studio/customize-workspace
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
      pkgs.htop
      pkgs.gnumake
      pkgs.vim
      # pkgs.texliveFull
      pkgs.python310
      pkgs.bibtex-tidy
  ];

  # Sets environment variables in the workspace
  env = {};
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      # "vscodevim.vim"
      # "adamraichu.pdf-viewer"
    ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        web = {
          # command to run the python http server on the specific port and bind to all interfaces
          command = [
            "python3.10"
            "-m"
            "http.server"
            "$PORT"
            "--bind"
            "0.0.0.0"
          ];
          manager = "web";
        };
      };
    };

    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        # Example: install JS dependencies from NPM
        # npm-install = "npm install";
      };
      # Runs when the workspace is (re)started
      onStart = {
        # Example: start a background task to watch and re-build backend code
        # watch-backend = "npm run watch-backend";
      };
    };
  };
}
