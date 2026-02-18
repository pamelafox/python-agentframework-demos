"""Un script simple para ver todas las sesiones y mensajes guardados en SQLite, con formato usando Rich."""

import argparse
import json
import sqlite3
import sys

from rich import print
from rich.console import Group
from rich.panel import Panel

DB_PATH = "chat_history.sqlite3"

parser = argparse.ArgumentParser(description="Ver sesiones y mensajes en la base de datos SQLite del historial de chat.")
parser.add_argument(
    "--db",
    default=DB_PATH,
    help="Ruta a la base de datos SQLite (por defecto: chat_history.sqlite3)",
)
parser.add_argument(
    "--values",
    action="store_true",
    help="Mostrar mensajes de cada sesión (por defecto: solo listar sesiones)",
)
args = parser.parse_args()

try:
    conn = sqlite3.connect(args.db)
    # Verificar que la tabla exista
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    if not cursor.fetchone():
        print(f"[red]No se encontró la tabla 'messages' en {args.db}[/red]")
        sys.exit(1)
except sqlite3.Error as e:
    print(f"[red]No se puede abrir la base de datos {args.db}: {e}[/red]")
    sys.exit(1)

# Obtener todas las sesiones con conteo de mensajes
sessions = conn.execute(
    "SELECT session_id, COUNT(*) as count FROM messages GROUP BY session_id ORDER BY session_id"
).fetchall()

if not sessions:
    print("[dim]No se encontraron sesiones en la base de datos.[/dim]")
    sys.exit(0)

print(f"\n[bold]Se encontraron {len(sessions)} sesión(es) en {args.db}[/bold]\n")

if not args.values:
    for session_id, count in sessions:
        print(f"  [bold cyan]{session_id}[/bold cyan] [dim]({count} mensajes)[/dim]")
    print()
    sys.exit(0)

for session_id, count in sessions:
    rows = conn.execute(
        "SELECT message_json FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()

    panels = []
    for i, (message_json,) in enumerate(rows):
        try:
            parsed = json.loads(message_json)
            role = parsed.get("role", {}).get("value", "unknown")
            contents = parsed.get("contents", [])
            # Extraer texto para mostrar
            parts = []
            for c in contents:
                if c.get("type") == "text":
                    parts.append(c["text"])
                elif c.get("type") == "function_call":
                    parts.append(f"[llamada de herramienta] {c['name']}({c['arguments']})")
                elif c.get("type") == "function_result":
                    parts.append(f"[resultado de herramienta] {c['result']}")
            display = "\n".join(parts) if parts else json.dumps(parsed, indent=2)
            color = {"user": "blue", "assistant": "green", "tool": "yellow"}.get(role, "white")
            panels.append(
                Panel(display, title=f"[bold {color}]{role}[/bold {color}] [dim]({i + 1}/{count})[/dim]")
            )
        except json.JSONDecodeError:
            panels.append(Panel(message_json, title=f"[dim]elemento {i + 1}/{count}[/dim]"))

    print(
        Panel(
            Group(*panels),
            title=f"[bold cyan]{session_id}[/bold cyan]",
            subtitle=f"[dim]{count} mensaje(s)[/dim]",
        )
    )
    print()

conn.close()
