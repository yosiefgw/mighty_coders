# How we work

- `main` is the stable branch. `dev` is where work is combined. Nobody pushes directly to either.
- One branch per task: `feature/<name>-<topic>` or `fix/<topic>`. Open an issue first for anything bigger than a small fix.
- Open a pull request into `dev`. At least one other team member reviews it with real comments before it is merged.
- Every PR description says what changed, how to test it, and which AI tool (if any) was used.
- Merge `dev` into `main` by pull request only, when everything runs from a clean start.
- Commit messages: short imperative first line.
