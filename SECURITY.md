# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

As a pre-1.0 project, only the latest minor release receives security updates. We recommend always upgrading to the latest version.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use [GitHub Security Advisories](https://github.com/oraios/synaptiq/security/advisories/new) to report vulnerabilities privately.

Please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Any potential impact

## Response Process

- **Acknowledgment**: We will acknowledge receipt within **48 hours**.
- **Assessment**: We will assess severity and affected versions within **7 days**.
- **Fix**: We aim to release a patch within **30 days** for confirmed vulnerabilities.
- **Disclosure**: We will coordinate disclosure timing with the reporter. We follow responsible disclosure practices and will credit reporters unless they prefer to remain anonymous.

## Scope

The following are in scope for security reports:

- The `synaptiq` Python package (published on PyPI)
- The MCP server and CLI tools
- Graph database storage and query handling
- File system access and `.gitignore` handling during ingestion

The following are out of scope:

- Vulnerabilities in upstream dependencies (report these to the relevant project)
- Issues requiring physical access to the machine running Synaptiq
- Social engineering attacks
