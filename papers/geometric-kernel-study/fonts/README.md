# Font Notes

This repository ships only redistributable fallback fonts.

The current UofSC brand guide specifies licensed Berlingske fonts for primary use. Those files are not bundled here. If the official fonts are already installed on your machine, `beamerthemeuofsc.sty` will prefer them automatically. Otherwise the theme uses the bundled open fonts below.

Bundled font families:

- `Oswald-Bold.ttf`, `Oswald-Regular.ttf`
  - Role: title and heading fallback
  - Upstream project: <https://github.com/googlefonts/OswaldFont>
  - License: SIL Open Font License 1.1
- `Arimo-Regular.ttf`, `Arimo-Bold.ttf`, `Arimo-Italic.ttf`, `Arimo-BoldItalic.ttf`
  - Role: sans-serif body fallback
  - Upstream project: <https://github.com/TypeNetwork/Arimo>
  - License: Apache License 2.0
- `Tinos-Regular.ttf`, `Tinos-Bold.ttf`, `Tinos-Italic.ttf`, `Tinos-BoldItalic.ttf`
  - Role: serif fallback
  - Upstream project: <https://github.com/googlefonts/tinos>
  - License: Apache License 2.0

See `LICENSE-OFL-1.1.txt` and `LICENSE-Apache-2.0.txt` in this directory for the bundled font license texts.
