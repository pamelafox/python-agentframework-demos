"""Un script simple para ver todas las claves y valores guardados en Redis, con formato usando Rich."""

import argparse
import json
import os
import sys

import redis
from dotenv import load_dotenv
from rich import print
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

load_dotenv(override=True)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

parser = argparse.ArgumentParser(description="Ver claves y valores en Redis.")
parser.add_argument(
    "--values",
    action="store_true",
    help="Mostrar valores de cada clave (por defecto: solo listar claves)",
)
args = parser.parse_args()

r = redis.from_url(REDIS_URL)

try:
    r.ping()
except redis.ConnectionError:
    print(f"[red]No se puede conectar a Redis en {REDIS_URL}[/red]")
    sys.exit(1)

keys = sorted(k.decode() for k in r.keys("*"))

if not keys:
    print("[dim]No se encontraron claves en Redis.[/dim]")
    sys.exit(0)

print(f"\n[bold]Se encontraron {len(keys)} clave(s) en Redis[/bold]\n")

if not args.values:
    for key in keys:
        key_type = r.type(key).decode()
        print(f"  [bold cyan]{key}[/bold cyan] [dim]({key_type})[/dim]")
    print()
    sys.exit(0)

for key in keys:
    key_type = r.type(key).decode()

    if key_type == "string":
        raw = r.get(key).decode()
        try:
            formatted = json.dumps(json.loads(raw), indent=2)
            content = Syntax(formatted, "json", theme="monokai", word_wrap=True)
        except json.JSONDecodeError:
            content = Text(raw)
        print(Panel(content, title=f"[bold cyan]{key}[/bold cyan] [dim]({key_type})[/dim]"))

    elif key_type == "list":
        items = r.lrange(key, 0, -1)
        panels = []
        for i, item in enumerate(items):
            raw = item.decode()
            try:
                parsed = json.loads(raw)
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
                    Panel(display, title=f"[bold {color}]{role}[/bold {color}] [dim]({i + 1}/{len(items)})[/dim]")
                )
            except json.JSONDecodeError:
                panels.append(Panel(raw, title=f"[dim]elemento {i + 1}/{len(items)}[/dim]"))
        print(
            Panel(
                Group(*panels),
                title=f"[bold cyan]{key}[/bold cyan] [dim]({key_type})[/dim]",
                subtitle=f"[dim]{len(items)} mensaje(s)[/dim]",
            )
        )

    elif key_type == "hash":
        fields = r.hgetall(key)
        lines = []
        for field, val in fields.items():
            lines.append(f"[bold]{field.decode()}[/bold]: {val.decode()}")
        print(
            Panel("\n".join(lines), title=f"[bold cyan]{key}[/bold cyan] [dim]({key_type})[/dim]")
        )

    elif key_type == "set":
        members = sorted(m.decode() for m in r.smembers(key))
        print(
            Panel(
                "\n".join(members), title=f"[bold cyan]{key}[/bold cyan] [dim]({key_type})[/dim]"
            )
        )

    elif key_type == "zset":
        items = r.zrange(key, 0, -1, withscores=True)
        lines = [f"{m.decode()}: {s}" for m, s in items]
        print(
            Panel("\n".join(lines), title=f"[bold cyan]{key}[/bold cyan] [dim]({key_type})[/dim]")
        )

    else:
        print(Panel(f"[dim]Tipo no soportado: {key_type}[/dim]", title=f"[bold cyan]{key}[/bold cyan] [dim]({key_type})[/dim]"))

    print()
