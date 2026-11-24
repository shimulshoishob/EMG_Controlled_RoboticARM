<div align="center">

# 📖 Documentation

**Everything about how the EMG Robotic Arm is built, connected and trained**

[↑ Project README](../README.md)

</div>

| Read this | When you want to know… |
|---|---|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | the layers, the threads, the full pathway from muscle to motion, control modes, the safety layers and where data is stored |
| **[NETWORK.md](NETWORK.md)** | how the armband and the laptop find each other, authenticate, stream and recover; ports, packet format, troubleshooting |
| **[MACHINE_LEARNING.md](MACHINE_LEARNING.md)** | data formats, windowing, filtering, the 204 features, all 25 models, fair evaluation, the exported bundle, latest results |
| **[CHANGES_AND_ARCHITECTURE.md](CHANGES_AND_ARCHITECTURE.md)** | engineering history: issues found in the original code, what changed, what was *not* verified on hardware |

## Reading order

1. New to the project → [Project README](../README.md) → [ARCHITECTURE](ARCHITECTURE.md)
2. Armband will not connect → [NETWORK](NETWORK.md#7-troubleshooting)
3. Training or choosing a model → [MACHINE_LEARNING](MACHINE_LEARNING.md) → [Model_Training](../Model_Training/README.md)
4. Safety review → [ARCHITECTURE § 6](ARCHITECTURE.md#6-safety-layer-by-layer) and the [hardware checklist](../README.md#-hardware-safety-checklist)

## Folder guides

| Folder | README |
|---|---|
| application | [src/emg_arm](../src/emg_arm/README.md) · [armband](../src/emg_arm/armband/README.md) · [arm](../src/emg_arm/arm/README.md) · [control](../src/emg_arm/control/README.md) · [ml](../src/emg_arm/ml/README.md) · [ui](../src/emg_arm/ui/README.md) |
| training | [Model_Training](../Model_Training/README.md) |
| hardware | [firmware](../firmware/README.md) · [config](../config/README.md) · [tools](../tools/README.md) |
| quality | [tests](../tests/README.md) |
| data | [recordings](../recordings/README.md) · [logs](../logs/README.md) |
