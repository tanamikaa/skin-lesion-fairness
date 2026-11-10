# How we work

## Branches
- `main` is protected — **no direct pushes**. Everything goes through a Pull Request.
- Branch naming: `<name>/<short-task>` e.g. `afzal/data-splits`, `anamika/focal-loss`,
  `khushi/vit-model`, `karanpal/fairness-metrics`.

## Daily flow
```bash
git checkout main
git pull
git checkout -b yourname/task-name
# ... work ...
git add <files>
git commit -m "data: add lesion-level stratified split"
git push -u origin yourname/task-name
# open a PR on GitHub, link the issue ("Closes #12"), request your reviewer
```

## Pull requests
- One PR = one task/issue. Keep it small.
- At least **1 approval** from your assigned reviewer (see `docs/TEAM_ROLES.md`) before merge.
- Run `pytest` locally before opening the PR.
- Use "Squash and merge".

## Issues & project board
- Every task in `docs/TEAM_ROLES.md` is a GitHub Issue, assigned to its owner.
- Labels: `data`, `cnn`, `vit`, `evaluation`, `explainability`, `report`, `bug`.
- Move cards on the Project board: Todo → In progress → In review → Done.

## Commit messages
`<area>: <what changed>` — areas: `data`, `model`, `train`, `eval`, `explain`, `docs`, `exp`.

## Rules
- **Never commit data, checkpoints (`*.pt`) or large files.** Share checkpoints via the team Drive folder.
- Notebooks are for exploration; move reusable code into `src/skinfair/`. Clear outputs before committing.
- Record every experiment in `results/EXPERIMENTS.md` (config, seed, metric, who ran it).
- Fix seeds (`seed` in config) so results are reproducible.
