# Release Process

This document describes the release process for the Amari mathematical computing library.

## 📋 Prerequisites

Before releasing, ensure you have:

1. **NPM Account**: Create an account at [npmjs.com](https://www.npmjs.com)
2. **NPM Token**: Generate an automation token for CI/CD
3. **Crates.io Account**: Create an account at [crates.io](https://crates.io)
4. **Repository Secrets**: Configure the following GitHub secrets:
   - `NPM_TOKEN`: Your npm automation token
   - `CRATES_TOKEN`: Your crates.io API token

## 🔄 Automated Release Process

The release process is fully automated through GitHub Actions. There are two ways to trigger a release:

### Option 1: Tag-based Release

```bash
# Update version in Cargo.toml files
# Commit changes
git add .
git commit -m "chore: bump version to 0.1.0"

# Create and push a version tag
git tag v0.1.0
git push origin v0.1.0
```

The workflow will automatically:
1. Build WASM packages for all targets (web, node, bundler)
2. Publish to npm registry
3. Create GitHub release with artifacts
4. Publish Rust crates to crates.io

### Option 2: Manual Workflow Dispatch

1. Go to Actions tab in GitHub
2. Select "Release" workflow
3. Click "Run workflow"
4. Enter version number (e.g., "0.1.0")
5. Click "Run workflow"

## 🏗️ Local Build Process

To build and test the npm package locally:

```bash
# Install wasm-pack if not already installed
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Navigate to WASM crate
cd amari-wasm

# Build for web target
wasm-pack build --target web --out-dir pkg --scope amari

# Build for Node.js
wasm-pack build --target nodejs --out-dir pkg-node --scope amari

# Build for bundlers (webpack, etc.)
wasm-pack build --target bundler --out-dir pkg-bundler --scope amari

# Test the package locally
cd pkg
npm link
cd ../../examples/typescript
npm link @amari/core
npm run dev
```

## 📦 Package Structure

After building, the package structure will be:

```
amari-wasm/
├── pkg/                    # Web target
│   ├── package.json
│   ├── amari_wasm.js
│   ├── amari_wasm_bg.wasm
│   ├── amari_wasm.d.ts
│   └── README.md
├── pkg-node/              # Node.js target
│   └── ...
└── pkg-bundler/           # Bundler target
    └── ...
```

## 🔍 Version Management

The project uses workspace versioning. Update versions in:

1. Root `Cargo.toml` (workspace version)
2. `amari-wasm/package.json`
3. Individual crate `Cargo.toml` files if needed

## 🧪 Pre-release Checklist

Before releasing, ensure:

- [ ] All tests pass: `cargo test --workspace`
- [ ] Clippy passes: `cargo clippy --workspace -- -D warnings`
- [ ] Format check: `cargo fmt --all -- --check`
- [ ] WASM builds: `cd amari-wasm && wasm-pack build`
- [ ] Examples work: Test TypeScript examples
- [ ] Documentation updated: README, CHANGELOG, API docs
- [ ] Version numbers updated consistently

## 📊 Release Targets

The release process publishes to multiple targets:

### NPM Package (@amari/core)
- **Web**: ES modules for browsers
- **Node**: CommonJS for Node.js
- **Bundler**: For webpack, rollup, etc.

### Crates.io Packages
- `amari-core`: Core geometric algebra
- `amari-tropical`: Tropical algebra
- `amari-dual`: Automatic differentiation
- `amari-info-geom`: Information geometry
- `amari-gpu`: WebGPU acceleration
- `amari-fusion`: Mathematical fusion
- `amari-wasm`: WASM bindings
- `amari-automata`: Cellular automata

## 🐛 Troubleshooting

### WASM Build Fails
```bash
# Clear cargo cache
cargo clean

# Reinstall wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Ensure correct Rust target
rustup target add wasm32-unknown-unknown
```

### NPM Publish Fails
```bash
# Check npm login
npm whoami

# Verify package name is available
npm view @amari/core

# Dry run
npm publish --dry-run
```

### GitHub Actions Fails
- Check secrets are configured correctly
- Verify permissions for GITHUB_TOKEN
- Check workflow syntax with act (local testing)

## 📈 Post-release

After a successful release:

1. **Announce**: Update project README with new version badge
2. **Document**: Update CHANGELOG.md with release notes
3. **Examples**: Update example dependencies to latest version
4. **Social**: Announce on Twitter, Reddit, Discord, etc.
5. **Monitor**: Check npm/crates.io download stats

## 🔒 Security

- Never commit tokens or secrets
- Use GitHub secrets for CI/CD
- Rotate tokens regularly
- Use 2FA on npm and crates.io accounts

## 📚 Resources

- [wasm-pack documentation](https://rustwasm.github.io/wasm-pack/)
- [npm publishing docs](https://docs.npmjs.com/cli/v8/commands/npm-publish)
- [crates.io publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [GitHub Actions documentation](https://docs.github.com/en/actions)

---

For questions or issues with the release process, please open an issue on GitHub.