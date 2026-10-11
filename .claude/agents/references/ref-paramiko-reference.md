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

SSHClient — connect(), exec_command(), invoke_shell(), open_sftp(); Transport — request_port_forward(), open_channel(); SFTPClient — get(), put(), listdir(), stat(); RSAKey / Ed25519Key — from_private_key_file(); AuthenticationException, SSHException for error handling

<!-- Document the primary classes, functions, or APIs that project code depends on from paramiko. -->

## Common Patterns & Pitfalls

Use SSHClient with AutoAddPolicy only in trusted environments; prefer RejectPolicy and known_hosts in production. Always close connections with client.close() or use context manager. For port forwarding, use transport.request_port_forward() and handle incoming channels in a thread. Pitfall: exec_command() stdout is blocking — read stdout before stderr to avoid deadlocks. Pitfall: set timeout= on connect() to avoid hanging on unreachable hosts.

<!-- Document common usage patterns, best practices, and known issues for paramiko 2.8.1. -->

## Key Conventions

- Follow project style rules when using paramiko
- Refer to authority sources for API contract accuracy
- Validate changes against existing tests before committing

## Related Agents

- `@technical-validator` — verify technical accuracy of paramiko usage
- `@primary-producer` — implements code that depends on paramiko
<!-- AGENTTEAMS:END content -->
