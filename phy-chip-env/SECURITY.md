# Security model

`phy-chip-env` executes model-generated ngspice decks, so every deck is treated
as untrusted input.

## Defenses in v0.1

Before starting a subprocess, the static gate:

- allows only an explicit set of dotcards and `.control` commands;
- rejects `.include`, `.lib`, `.load`, `.osdi`, `.source`, `.shell`, writes,
  redirection, command chaining, malformed markers, and external file fields;
- rejects nested or unterminated control blocks;
- requires a device and an analysis before the simulator can pass.

The subprocess runner:

- creates a new mode-0700 temporary directory per rollout;
- sets `HOME` and `TMPDIR` to that directory and supplies a minimal environment;
- writes a mode-0400 deck and bounded output files;
- applies CPU, address-space/data, output-file-size, file-descriptor, process,
  and core-dump resource limits;
- uses a wall-clock timeout and a separate process session;
- drops supplementary groups and switches to `nobody` when the parent is root;
- deletes the temporary directory after the result is read.

## Non-goals and residual risk

This is process/resource isolation, not a complete kernel security boundary.
It does not install seccomp filters or create network, PID, user, mount, or
filesystem namespaces. On a non-root host, ngspice runs with the invoking user's
identity. A vulnerability in ngspice itself could therefore exceed the intended
SPICE-language boundary.

For untrusted multi-tenant traffic, run the package in the supplied container
under an additional runtime policy: no network, read-only root filesystem,
non-root user, dropped Linux capabilities, and strict CPU/memory limits.

## Reporting

Do not include private model outputs or credentials in a report. Open a minimal
reproducer through the repository issue tracker and mark security-sensitive
details clearly.
