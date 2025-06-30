from datalake.enriched.poetry import Poetry

def run():
    source_name = "poetry"

    Poetry().run(destination_path=f"datalake/bucket/enriched/{source_name}")


if __name__ == "__main__":
    run()
