<!-- AGENTTEAMS:BEGIN content v=1 -->
# paramiko Reference — ProjectRepositories

> Quick-reference for **paramiko 2.8.1** (library) in ProjectRepositories.
> This is a lightweight reference file, not an agent. For operational procedures, consult the tool's reference/skill document, or escalate to `@orchestrator`.

---

## Version

`paramiko` `2.8.1`

## Configuration

**Config files:** `N/A`

## Official Documentation

https://docs.paramiko.org/en/stable/api/

## Key API Surface

SSHClient.connect/exec_command/invoke_shell/open_sftp, Transport.request_port_forward, SFTPClient.get/put, RSAKey/Ed25519Key.from_private_key_file

<!-- Document the primary classes, functions, or APIs that project code depends on from paramiko. -->

## Common Patterns & Pitfalls

Use AutoAddPolicy only in trusted environments; prefer RejectPolicy in production. Always close connections. Set timeout= on connect() to avoid hangs on unreachable hosts.

<!-- Document common usage patterns, best practices, and known issues for paramiko 2.8.1. -->

## Key Conventions

- Follow project style rules when using paramiko
- Refer to authority sources for API contract accuracy
- Validate changes against existing tests before committing

## Related Agents

- `@technical-validator` — verify technical accuracy of paramiko usage
- `@primary-producer` — implements code that depends on paramiko
<!-- AGENTTEAMS:END content -->
