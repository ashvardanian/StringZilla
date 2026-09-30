/**
 *  @file javascript/stage-prebuild.cjs
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Copies the addon `cmake-js` produced to where the loaders find it.
 *
 *  `prebuilds/<platform>-<arch>/node.napi.node` is where `node-gyp-build` resolves it, and in a
 *  checkout the `@stringzilla/<platform>-<arch>` package the release publishes gets a copy too.
 *  A `.cjs`, since the package's `"type": "module"` would read a `.js` as an ES module.
 *  Usage: `node javascript/stage-prebuild.cjs [build directory] [arch]`.
 */
const fs = require("fs");
const path = require("path");

const [, , buildDirectory = "build_node", architecture = process.arch] = process.argv;
const source = path.join(buildDirectory, "Release", "stringzilla.node");
const platform = `${process.platform}-${architecture}`;
const targets = [path.join("prebuilds", platform, "node.napi.node")];
const platformPackage = path.join("javascript", `@stringzilla-${platform}`);
if (fs.existsSync(platformPackage)) targets.push(path.join(platformPackage, "stringzilla.node"));

for (const target of targets) {
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.copyFileSync(source, target);
    console.log(`staged ${source} -> ${target}`);
}
