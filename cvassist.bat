@echo off
rem Shortcut: "cvassist help", "cvassist ask ...". Uses the project's .venv if it exists, otherwise the system Python.
set "HERE=%~dp0"
if exist "%HERE%.venv\Scripts\python.exe" (
    "%HERE%.venv\Scripts\python.exe" "%HERE%cvassist.py" %*
) else (
    python "%HERE%cvassist.py" %*
)
