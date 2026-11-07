# EMG Mouse: starter

Two people, each running a Claude Code agent in the Claude desktop app on their own computer and account,
build this in parallel. **GitHub is the only link between the two agents.**

- `CLAUDE.md`: shared context and rules (every Claude Code session here reads it automatically)
- `docs/INTERFACE.md`: the contract between frontend and backend
- `docs/HARDWARE.md`: the sEMG circuit, IMUs and camera
- `docs/handoff/FROM_FRONTEND.md`, `FROM_BACKEND.md`: message boards (each side writes only its own)
- `prompts/FRONTEND.md`, `prompts/BACKEND.md`: role prompts

## Kickoff message for FRONTEND (Jerry's computer)
> You are the FRONTEND agent. Read prompts/FRONTEND.md and follow it exactly. Read every file it lists, then reply
> with your plan for M1–M2 and any questions for me. Don't write code until I approve.

## Kickoff message for BACKEND (teammate's computer)
> You are the BACKEND agent. Read prompts/BACKEND.md and follow it exactly. Read every file it lists, then reply
> with your plan for M1–M3 and any questions for me. Don't write code until I approve.

## Sync message (send whenever the other person says they pushed)
> Run git pull --rebase, read the other side's handoff file for new messages to you, and act on them.
