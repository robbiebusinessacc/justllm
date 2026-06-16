# Security Policy

## Supported versions

justllm is pre-1.0; security fixes land on the latest released version.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

- Preferred: open a private [security advisory](https://github.com/robbiebusinessacc/justllm/security/advisories/new).
- Or email: robbie.github@gmail.com

You'll get an acknowledgement as soon as possible, and a fix or mitigation plan
once the report is confirmed. Responsible disclosure is appreciated.

## A note on credentials

justllm never logs or transmits your API keys — it passes them straight to the
underlying provider SDK via LiteLLM. Keep keys in environment variables; never
commit them or paste them into issues, PRs, or chats.
