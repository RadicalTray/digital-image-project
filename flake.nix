{
  description = "";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, ... }@inputs:
    let
      supportedSystems = [
        "x86_64-linux"
        # "aarch64-linux"
        # "x86_64-darwin"
        # "aarch64-darwin"
      ];

      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python314
            ] ++ (with pkgs.python314Packages; [
              python-lsp-server
              numpy
              pandas
              torch
              torchvision
              matplotlib
              pillow
              opencv4
            ]);
          };
          tensorflow = pkgs.mkShell {
            packages = with pkgs; [
              python313
            ] ++ (with pkgs.python313Packages; [
              python-lsp-server
              numpy
              pandas
              torch
              torchvision
              matplotlib
              pillow
              opencv4
              tensorflow
            ]);
          };
        }
      );
    };
}

