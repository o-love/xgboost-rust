{
  description = "xgboost-rust dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        rustc cargo
        clang
        llvmPackages.libclang
        llvmPackages.libllvm
        pkg-config
      ];

      # bindgen looks here for libclang.so
      LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

      # Often helps when crates call out to cc / linkers
      BINDGEN_EXTRA_CLANG_ARGS = [
        "-isystem" "${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.llvmPackages.libclang.version}/include"
      ];
    };
  };
}
