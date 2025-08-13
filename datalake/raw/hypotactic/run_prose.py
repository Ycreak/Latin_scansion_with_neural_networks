from datalake.raw.hypotactic.prose import Prose


def run():
    source_name = "hypotactic"

    Prose().run(
        source_path=f"datalake/bucket/landing_zone/{source_name}",
        destination_path=f"datalake/bucket/raw/{source_name}",
    )


if __name__ == "__main__":
    run()
