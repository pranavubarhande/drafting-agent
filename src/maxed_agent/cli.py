import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def status() -> None:
    console.print("maxed-agent scaffold OK. Generate, verify, learn, and eval commands land here next.")


if __name__ == "__main__":
    app()
