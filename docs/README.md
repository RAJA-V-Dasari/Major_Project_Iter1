# Documentation

The repo root [`README.md`](../README.md) is the overview. The files here
cover what spans both halves; each half documents itself beside its code.

| file | answers |
|---|---|
| [`PIPELINE.md`](PIPELINE.md) | What do I run, in what order, what does each stage read and write, and what invalidates what? |
| [`DATA.md`](DATA.md) | Where does the data come from, where does it live, and what are the rules? |
| [`HANDOFF.md`](HANDOFF.md) | What exactly does the reading half give the marking half? |
| [`slides/`](slides/) | The two review decks |
| [`history/`](history/) | The marking half's git history from before it joined this repo (a git bundle) |

Beside the code:

| folder | documents |
|---|---|
| [`../reading/README.md`](../reading/README.md), [`../reading/docs/`](../reading/docs/) | how pages become booklets; what was measured (`DONE.md`), what is left (`TODO.md`), the batch00 review, the geometry module, the dataset card |
| [`../marking/README.md`](../marking/README.md), [`../marking/docs/`](../marking/docs/) | how booklets are marked; architecture, run order, data formats, glossary, local model setup; known weaknesses in `IMPROVEMENTS.md` |
| [`../CLAUDE.md`](../CLAUDE.md) | conventions for working on the code |
