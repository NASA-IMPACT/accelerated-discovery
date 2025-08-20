# Integration Testing Branch

## Overview

The `integration` branch serves as a continuous integration testing environment that automatically merges all feature and refactor PRs to test their compatibility together.

## Branch Strategy

- **Base**: Always ahead of `develop`
- **Purpose**: Test multiple merged PRs together before they reach production
- **Automation**: Automatically syncs when PRs are merged to `develop`

## Workflow

1. **Feature Development**: Create feature branches from `develop`
2. **PR Creation**: Open PRs targeting `develop`
3. **Auto-Integration**: GitHub Actions automatically merges the PR branch into `integration`
4. **Integration Testing**: Run comprehensive tests on `integration` branch with the new feature
5. **Review & Merge**: After approval and successful integration testing, merge PR to `develop`
6. **Conflict Resolution**: If merge conflicts occur, an issue is automatically created

## Automation Details

- **Trigger**: When PRs are opened, updated, or reopened targeting `develop`
- **Action**: `.github/workflows/integration-branch-sync.yml`
- **Merge Strategy**: No-fast-forward merge to preserve commit history
- **Conflict Handling**: Creates GitHub issue with resolution instructions

## Manual Conflict Resolution

If the automated merge fails:

```bash
git checkout integration
git fetch origin
git merge origin/develop
# Resolve any conflicts
git add .
git commit -m "Resolve merge conflicts"
git push origin integration
```

## Best Practices

- Never directly commit to `integration`
- Use `integration` for testing multi-feature compatibility
- Monitor GitHub Actions for merge failures
- Address conflicts promptly when issues are created
- Keep `integration` as close to `develop` as possible

## Testing on Integration

Use the `integration` branch to:
- Run full test suites with multiple features
- Test feature interactions
- Validate deployment configurations
- Perform load testing with combined changes