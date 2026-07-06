<div align="center">
  <h1>🤖 Dexmate Robot Control and Sensing API</h1>
</div>

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

## 📦 Installation

```shell
pip install dexcontrol
```

To run the examples in this repo, you can try:

```shell
pip install dexcontrol[example]
```

## ⚠️ Version Compatibility

**Important:** `dexcontrol >= 0.5.0` requires robot firmware `>= 0.5.0`. Older firmware will not work with this release — update the firmware first, or pin `dexcontrol` to `0.4.x` until you can upgrade.

If your firmware is outdated, please update it before installing the new version to ensure full compatibility. Please contact the Dexmate team if you do not know how to do it.

### Firmware Compatibility Matrix

`dexcontrol` and the robot firmware are released in lockstep on the minor version: a given `dexcontrol 0.X.y` is compatible with firmware `0.X.*`. You can check the firmware version on your robot by running `examples/troubleshooting/display_robot_info.py`.

| `dexcontrol` version | Compatible firmware version |
| --- | --- |
| `0.5.x` | `0.5.x` |
| `0.4.x` | `0.4.x` |
| `0.3.x` | `0.3.x` |
| `0.2.x` | `0.2.x` |

If your firmware minor version does not match the `dexcontrol` version you intend to install, the client will print a version warning at startup and some features may not work correctly. Update the firmware first (see [vega-firmware](https://github.com/dexmate-ai/vega-firmware)) or pin `dexcontrol` to a version compatible with your current firmware.

**📋 See [CHANGELOG.md](./CHANGELOG.md) for detailed release notes and version history.**

## 📄 Licensing

This project is **dual-licensed**:

### 🔓 Open Source License
This software is available under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
See the [LICENSE](./LICENSE) file for details.

### 💼 Commercial License
For businesses that want to use this software in proprietary applications without the AGPL requirements, commercial licenses are available.

**📧 Contact us for commercial licensing:** contact@dexmate.ai

**Commercial licenses provide:**
- ✅ Right to use in closed-source applications
- ✅ No source code disclosure requirements
- ✅ Priority support options


## 📚 Examples

Explore our comprehensive examples in the `examples/` directory:

- 🎮 **Basic Control** - Simple movement and sensor reading
- 🎯 **Advanced Control** - Complex manipulation tasks
- 📺 **Teleoperation** - Remote control interfaces
- 🔧 **Troubleshooting** - Diagnostic and maintenance tools

---

<div align="center">
  <h3>🤝 Ready to build amazing robots?</h3>
  <p>
    <a href="mailto:contact@dexmate.ai">📧 Contact Us</a> •
    <a href="./examples/">📚 View Examples</a> •
  </p>
</div>
