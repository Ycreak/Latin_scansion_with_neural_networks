from lsnn.raw.hypotactic.prose import Prose

def run():
    source_name = "hypotactic"
    
    Prose().run(source_path=f"lsnn/bucket/landing_zone/{source_name}", destination_path=f"lsnn/bucket/raw/{source_name}")


if __name__ == "__main__":
    run()
