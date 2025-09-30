# GitHub Actions CI Setup for Mathematical Correctness

## Overview

This repository uses both **local pre-commit hooks** and **GitHub Actions** to ensure mathematical correctness. This dual approach provides:

- **Fast local feedback** via pre-commit hooks (~30 seconds)
- **Comprehensive cloud verification** via GitHub Actions (~5-8 minutes)
- **Parallel test execution** to minimize CI time
- **Cost optimization** for private repositories

## Workflows

### 1. `mathematical-correctness.yml`
**Purpose**: Comprehensive verification matching pre-commit hook
**Runtime**: ~5-7 minutes
**Triggers**: Push to main/develop, all PRs

Runs the complete test suite sequentially:
- Code formatting (rustfmt)
- Linting (clippy)
- Core mathematical tests
- Formal verification tests
- Documentation builds

### 2. `parallel-verification.yml`
**Purpose**: Optimized parallel testing
**Runtime**: ~3-5 minutes (parallel execution)
**Triggers**: Push to main/develop, all PRs

Parallel jobs:
- **Code Quality**: Format + Clippy (5 min)
- **Core Algebra**: Tests core crates in parallel (8 min)
- **Advanced Features**: Tests complex crates in parallel (10 min)
- **GPU Acceleration**: GPU tests with CI fallbacks (8 min)
- **Formal Verification**: Verification tests (8 min)
- **Documentation**: Doc builds (5 min)
- **Integration**: Final integration check (10 min)

### 3. `status-badges.yml`
**Purpose**: Generate dynamic status badges
**Runtime**: ~2 minutes
**Triggers**: Push to main, weekly schedule

Creates badges showing:
- Total test count (e.g., "579 passing")
- Verification count (e.g., "140 verified")
- Mathematical correctness status

## Cost Analysis

### Free Tier Usage (Private Repos)
- **Parallel workflow**: ~5 minutes per run
- **Monthly estimate**: ~150-200 commits × 5 min = 750-1000 minutes
- **Free tier**: 2,000 minutes/month
- **Conclusion**: Well within free limits

### Public Repos
- **Completely free** - unlimited GitHub Actions minutes

### Optimization Strategies

1. **Conditional execution**:
   ```yaml
   if: contains(github.event.head_commit.message, '[skip ci]') == false
   ```

2. **Path-based triggers**:
   ```yaml
   paths:
     - 'amari-*/**'
     - 'Cargo.toml'
     - 'Cargo.lock'
   ```

3. **Matrix parallelization**:
   ```yaml
   strategy:
     matrix:
       crate: [amari-core, amari-tropical, amari-dual]
   ```

## Recommended Setup

### For Most Projects
Use **both workflows**:
- `parallel-verification.yml` for regular commits (faster feedback)
- `mathematical-correctness.yml` for release branches (comprehensive check)

### For High-Frequency Development
Use **parallel-verification.yml** only with path triggers:
```yaml
on:
  push:
    paths:
      - 'amari-*/**'
      - '!**.md'  # Skip docs-only changes
```

### For Release Candidates
Add release-specific workflow:
```yaml
on:
  push:
    tags:
      - 'v*'
```

## Local vs. Cloud Testing

| Aspect | Pre-commit Hook | GitHub Actions |
|--------|----------------|----------------|
| **Speed** | 30 seconds | 5-8 minutes |
| **Coverage** | Full test suite | Full + parallel |
| **Environment** | Developer machine | Clean Ubuntu |
| **GPU Tests** | May work locally | Graceful fallback |
| **Cost** | Free | Free/cheap |
| **Timing** | Before commit | After push |

## Status Integration

Add to your README.md:
```markdown
![Mathematical Correctness](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/username/gist-id/raw/math-correctness.json)
![Tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/username/gist-id/raw/test-count.json)
![Formal Verification](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/username/gist-id/raw/verification.json)
```

## Mathematical Integrity Guarantee

The combination of pre-commit hooks + GitHub Actions ensures:

✅ **No regression commits**: Pre-commit hook blocks bad commits
✅ **Public verification**: GitHub Actions provides transparent testing
✅ **Parallel efficiency**: Matrix jobs minimize CI time
✅ **Cost effectiveness**: Optimized for free tier usage
✅ **Mathematical correctness**: 100% test coverage requirement