{
  description = "whisper.cpp with streaming and preview";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
  };

  outputs = { self, nixpkgs, flake-utils, crane }:
    let
      nixosModule = { config, lib, pkgs, ... }:
        let
          cfg = config.services.whisper-transcriber;

          instanceOpts = { name, config, ... }: {
            options = {
              enable = lib.mkEnableOption "this Whisper transcription service instance" // {
                default = true;
              };

              package = lib.mkOption {
                type = lib.types.package;
                default = self.packages.${pkgs.system}.transcriber;
                defaultText = lib.literalExpression "whisper-preview.packages.\${pkgs.system}.transcriber";
                description = ''
                  The whisper-transcriber package to use.
                  Use `whisper-preview.packages.$${pkgs.system}.transcriber-vulkan` for GPU acceleration.
                '';
              };

              address = lib.mkOption {
                type = lib.types.str;
                default = "127.0.0.1";
                description = "Address to bind to";
              };

              port = lib.mkOption {
                type = lib.types.port;
                description = "Port to listen on";
              };

              modelPath = lib.mkOption {
                type = lib.types.path;
                description = "Path to whisper model file";
              };

              tokenFile = lib.mkOption {
                type = lib.types.nullOr lib.types.path;
                default = null;
                description = ''
                  Path to API token file for authentication.
                  The file will be loaded via systemd LoadCredential for security.
                '';
              };

              beamSize = lib.mkOption {
                type = lib.types.nullOr lib.types.int;
                default = null;
                description = "Beam search size (mutually exclusive with bestOf)";
              };

              bestOf = lib.mkOption {
                type = lib.types.nullOr lib.types.int;
                default = 1;
                description = "Greedy search best-of count (mutually exclusive with beamSize)";
              };

              user = lib.mkOption {
                type = lib.types.str;
                default = "whisper-transcriber";
                description = "User to run the service as";
              };

              group = lib.mkOption {
                type = lib.types.str;
                default = "whisper-transcriber";
                description = "Group to run the service as";
              };

              dynamicAudioCtx = lib.mkOption {
                type = lib.types.bool;
                default = false;
                description = "Scale audio context to buffer length (faster for short audio)";
              };

              temperatureInc = lib.mkOption {
                type = lib.types.nullOr lib.types.float;
                default = null;
                description = "Temperature increment on decode failure (0 = no retries, whisper default: 0.2)";
              };

              entropyThold = lib.mkOption {
                type = lib.types.nullOr lib.types.float;
                default = null;
                description = "Entropy threshold for decode retry (whisper default: 2.4)";
              };

              enableGpu = lib.mkOption {
                type = lib.types.bool;
                default = false;
                description = ''
                  Enable GPU access for Vulkan acceleration.
                  This grants the service access to /dev/dri/* devices and adds the user to video/render groups.
                  Required when using transcriber-vulkan package.
                '';
              };
            };
          };
        in
        {
          options.services.whisper-transcriber = lib.mkOption {
            type = lib.types.attrsOf (lib.types.submodule instanceOpts);
            default = {};
            description = "Whisper transcription service instances";
          };

          config = let
            enabledInstances = lib.filterAttrs (_: cfg: cfg.enable) cfg;
            allUsers = lib.unique (lib.mapAttrsToList (_: cfg: cfg.user) enabledInstances);
            allGroups = lib.unique (lib.mapAttrsToList (_: cfg: cfg.group) enabledInstances);

            # Map users to their GPU enablement status
            userGpuMap = lib.listToAttrs (lib.mapAttrsToList (name: instanceCfg: {
              name = instanceCfg.user;
              value = instanceCfg.enableGpu;
            }) enabledInstances);
          in lib.mkIf (enabledInstances != {}) {
            assertions = lib.flatten (lib.mapAttrsToList (name: instanceCfg: [
              {
                assertion = instanceCfg.beamSize == null || instanceCfg.bestOf == null;
                message = "whisper-transcriber.${name}: beamSize and bestOf are mutually exclusive";
              }
            ]) enabledInstances);

            users.users = lib.genAttrs allUsers (user: {
              isSystemUser = true;
              group = user;
              description = "Whisper transcriber service user";
              # Add GPU groups if any instance using this user has GPU enabled
              extraGroups = lib.optional (userGpuMap.${user} or false) "video"
                ++ lib.optional (userGpuMap.${user} or false) "render";
            });

            users.groups = lib.genAttrs allGroups (_: {});

            systemd.services = lib.mapAttrs' (name: instanceCfg: lib.nameValuePair
              "whisper-transcriber-${name}"
              {
                description = "Whisper streaming transcription service (${name})";
                wants = [ "network-online.target" ];
                after = [ "network-online.target" ];
                wantedBy = [ "multi-user.target" ];

                serviceConfig = {
                  ExecStart = let
                    args = [
                      "${instanceCfg.package}/bin/transcriber"
                      "--address" instanceCfg.address
                      "--port" (toString instanceCfg.port)
                      "--model" (toString instanceCfg.modelPath)
                    ]
                    ++ lib.optionals (instanceCfg.tokenFile != null) [
                      "--token-file" "\${CREDENTIALS_DIRECTORY}/token"
                    ]
                    ++ lib.optionals (instanceCfg.beamSize != null && instanceCfg.bestOf == null) [
                      "--beam-size" (toString instanceCfg.beamSize)
                    ]
                    ++ lib.optionals (instanceCfg.bestOf != null && instanceCfg.beamSize == null) [
                      "--best-of" (toString instanceCfg.bestOf)
                    ]
                    ++ lib.optionals instanceCfg.dynamicAudioCtx [
                      "--dynamic-audio-ctx"
                    ]
                    ++ lib.optionals (instanceCfg.temperatureInc != null) [
                      "--temperature-inc" (toString instanceCfg.temperatureInc)
                    ]
                    ++ lib.optionals (instanceCfg.entropyThold != null) [
                      "--entropy-thold" (toString instanceCfg.entropyThold)
                    ];
                  in lib.escapeShellArgs args;

                  User = instanceCfg.user;
                  Group = instanceCfg.group;
                  Restart = "on-failure";
                  RestartSec = "5s";

                  # Load token via systemd credentials (more secure than passing via file)
                  LoadCredential = lib.optionals (instanceCfg.tokenFile != null) [
                    "token:${instanceCfg.tokenFile}"
                  ];

                  RuntimeDirectory = "whisper-transcriber-${name}";
                  RuntimeDirectoryMode = "0700";

                  # Security hardening (based on wyoming/faster-whisper.nix)
                  CapabilityBoundingSet = "";
                  DevicePolicy = "closed";
                  # Allow GPU device access when Vulkan is enabled
                  DeviceAllow = lib.optionals instanceCfg.enableGpu [
                    "/dev/dri/card0 rw"
                    "/dev/dri/renderD128 rw"
                  ];
                  LockPersonality = true;
                  MemoryDenyWriteExecute = true;
                  NoNewPrivileges = true;
                  # Disable PrivateDevices when GPU access is needed
                  PrivateDevices = !instanceCfg.enableGpu;
                  PrivateTmp = true;
                  PrivateUsers = !instanceCfg.enableGpu;
                  ProtectClock = true;
                  ProtectControlGroups = true;
                  ProtectHome = true;
                  ProtectHostname = true;
                  ProtectKernelLogs = true;
                  ProtectKernelModules = true;
                  ProtectKernelTunables = true;
                  ProtectProc = "invisible";
                  ProtectSystem = "strict";
                  ProcSubset = "pid";
                  RemoveIPC = true;
                  RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
                  RestrictNamespaces = true;
                  RestrictRealtime = true;
                  RestrictSUIDSGID = true;
                  SystemCallArchitectures = "native";
                  SystemCallFilter = [ "@system-service" "~@privileged" ];
                  UMask = "0077";
                };
              }
            ) enabledInstances;
          };
        };
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        craneLib = crane.mkLib pkgs;

        # Test fixtures: audio files
        jfk-audio = pkgs.fetchurl {
          url = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav";
          hash = "sha256-Wd+5pKyzb+Kir/wUusvuKSD/Q1yxPMMUoIwT9munhg4=";
        };
        bbc-russian-original = pkgs.fetchurl {
          url = "https://open.live.bbc.co.uk/mediaselector/6/redir/version/2.0/mediaset/audio-nondrm-download-low/proto/https/vpid/p0n0d8sl.mp3";
          hash = "sha256-fe4doCNVow5emV6UMvmdoKsmJOioen5IkjMmfA3Fs7A=";
        };
        bbc-russian-audio = pkgs.runCommand "bbc-russian.wav" {} ''
          ${pkgs.sox}/bin/sox ${bbc-russian-original} -r 16000 -c 1 -b 16 $out
        '';

        # Whisper models
        models = {
          tiny = pkgs.fetchurl {
            name = "ggml-tiny.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin";
            hash = "sha256-vgfgSOHlma1GNByNKhNWRQl6U4IhZ4t6zdGxkZxuGyE=";
          };

          base = pkgs.fetchurl {
            name = "ggml-base.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";
            hash = "sha256-YO1bw90U7qhWST0zQ0m0BXgt3K8AKNS130CINF+6Lv4=";
          };

          small = pkgs.fetchurl {
            name = "ggml-small.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin";
            hash = "sha256-G+OpsgY4Z7k35k4ux0gzZKeZF+FX+pjF2UtcH//qmHs=";
          };

          medium = pkgs.fetchurl {
            name = "ggml-medium.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin";
            hash = "sha256-1xFYUKQfj8m/pDVxJLYnYDQZmHvZmWWM0Rk9+O9FDSU=";
          };

          large-v3 = pkgs.fetchurl {
            name = "ggml-large-v3.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin";
            hash = "sha256-ZNGCtEC5jVIDxPm9VBVE2ExgUZbE97hF36EfsjWU0eI=";
          };

          large-v3-turbo = pkgs.fetchurl {
            name = "ggml-large-v3-turbo.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin";
            hash = "sha256-H8cPd0046xaZk6w5Huo1fvR8iHV+9y7llDh5t+jivGk=";
          };

          # Quantized variants (smaller file size, slightly lower quality)
          tiny-q5 = pkgs.fetchurl {
            name = "ggml-tiny-q5_1.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin";
            hash = "sha256-E0oCD8mF3yqaPxhFUT3p4pu3gYlPxWFe2M4XZjdOibY=";
          };

          base-q5 = pkgs.fetchurl {
            name = "ggml-base-q5_1.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin";
            hash = "sha256-Qi8a5FKt5vMKAE1+XGpDGV5EM7w3C/I/rJzFkfAaiJg=";
          };

          small-q5 = pkgs.fetchurl {
            name = "ggml-small-q5_1.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin";
            hash = "sha256-QIXr6XeBQ+NyLWx7K7h9IzPJKSp1E0LsOOD9q9JJFq0=";
          };

          medium-q5 = pkgs.fetchurl {
            name = "ggml-medium-q5_0.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin";
            hash = "sha256-PAznBJkV+hCvAuC0cT3KR/l5V2P0k3LcQ0AXjHB/lBM=";
          };

          large-v3-turbo-q5 = pkgs.fetchurl {
            name = "ggml-large-v3-turbo-q5_0.bin";
            url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin";
            hash = "sha256-OUIhcJzVrR9AxG5gMcphvOiJMebgiMGIKUxtWlX/p+I=";
          };
        };

        craneArgs = {
          src = craneLib.cleanCargoSource ./rust;
          strictDeps = true;

          nativeBuildInputs = with pkgs; [
            cmake
            git
            llvmPackages.clang
            llvmPackages.libclang
            pkg-config
            shaderc
          ];

          buildInputs = with pkgs; [
            libopus
            libpulseaudio
            vulkan-headers
            vulkan-loader
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        };
        cargoArtifacts = craneLib.buildDepsOnly craneArgs;
        workspace = craneLib.buildPackage (craneArgs // {
          inherit cargoArtifacts;
        });

        # Build with Vulkan GPU acceleration enabled
        cargoArtifactsVulkan = craneLib.buildDepsOnly (craneArgs // {
          cargoExtraArgs = "--features vulkan";
        });
        workspaceVulkan = craneLib.buildPackage (craneArgs // {
          cargoArtifacts = cargoArtifactsVulkan;
          cargoExtraArgs = "--features vulkan";
        });

      in
      {
        packages = {
          default = workspace;
          inherit workspace;
          transcriber = workspace;  # Alias for convenience
          transcriber-vulkan = workspaceVulkan;  # With GPU acceleration

          # Export all whisper models
          whisper-model-tiny = models.tiny;
          whisper-model-base = models.base;
          whisper-model-small = models.small;
          whisper-model-medium = models.medium;
          whisper-model-large-v3 = models.large-v3;
          whisper-model-large-v3-turbo = models.large-v3-turbo;

          # Quantized models
          whisper-model-tiny-q5 = models.tiny-q5;
          whisper-model-base-q5 = models.base-q5;
          whisper-model-small-q5 = models.small-q5;
          whisper-model-medium-q5 = models.medium-q5;
          whisper-model-large-v3-turbo-q5 = models.large-v3-turbo-q5;

          # Convenience package with all models
          whisper-models-all = pkgs.linkFarm "whisper-models-all" [
            { name = "tiny.bin"; path = models.tiny; }
            { name = "base.bin"; path = models.base; }
            { name = "small.bin"; path = models.small; }
            { name = "medium.bin"; path = models.medium; }
            { name = "large-v3.bin"; path = models.large-v3; }
            { name = "large-v3-turbo.bin"; path = models.large-v3-turbo; }
            { name = "tiny-q5.bin"; path = models.tiny-q5; }
            { name = "base-q5.bin"; path = models.base-q5; }
            { name = "small-q5.bin"; path = models.small-q5; }
            { name = "medium-q5.bin"; path = models.medium-q5; }
            { name = "large-v3-turbo-q5.bin"; path = models.large-v3-turbo-q5; }
          ];

          # Test fixtures (audio files + models for testing)
          fixtures = pkgs.linkFarm "whisper-fixtures" [
            { name = "jfk.wav"; path = jfk-audio; }
            { name = "bbc_russian.wav"; path = bbc-russian-audio; }
            { name = "ggml-tiny.bin"; path = models.tiny; }
            { name = "ggml-large-v3-turbo.bin"; path = models.large-v3-turbo; }
            { name = "ggml-large-v3-turbo-q5_0.bin"; path = models.large-v3-turbo-q5; }
          ];
        };

        apps = {
          transcriber = flake-utils.lib.mkApp {
            drv = workspace;
            exePath = "/bin/transcriber";
          };
          default = self.apps.${system}.transcriber;
        };

        devShells.default = craneLib.devShell {
          inputsFrom = [ workspace ];
          packages = with pkgs; [ rust-analyzer cargo-watch vulkan-tools ];
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.lib.getVersion pkgs.llvmPackages.clang}/include -isystem ${pkgs.glibc.dev}/include";
          shellHook = ''
            mkdir -p fixtures
            ln -sf ${jfk-audio} fixtures/jfk.wav
            ln -sf ${bbc-russian-audio} fixtures/bbc_russian.wav
            ln -sf ${models.tiny} fixtures/ggml-tiny.bin
            ln -sf ${models.large-v3-turbo} fixtures/ggml-large-v3-turbo.bin
            ln -sf ${models.large-v3-turbo-q5} fixtures/ggml-large-v3-turbo-q5_0.bin
          '';
        };
      }
    ) // {
      inherit nixosModule;
      nixosModules = {
        whisper-transcriber = nixosModule;
        default = nixosModule;
      };
    };
}
