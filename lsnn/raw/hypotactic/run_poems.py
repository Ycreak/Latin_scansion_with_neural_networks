from lsnn.raw.hypotactic.poems import Poems

def run():
    source_name = "hypotactic"
    
    Poems().run(source_path=f"lsnn/bucket/landing_zone/{source_name}", destination_path=f"lsnn/bucket/raw/{source_name}")


if __name__ == "__main__":
    run()
