from datalake.landing_zone.hypotactic.ingest_hypotactic import Hypotactic


def run():
    source_name = "hypotactic"

    Hypotactic().run(destination_path=f"datalake/bucket/landing_zone/{source_name}")


if __name__ == "__main__":
    run()
