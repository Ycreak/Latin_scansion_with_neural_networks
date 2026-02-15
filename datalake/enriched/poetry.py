import datalake.utilities as util
import polars as pl


class Poetry:
    """
    In the Poetry class we collect all lines from the cleaned layer and put them into one Polars dataframe.
    """

    def run(self, destination_path: str) -> None:
        raw_bucket_path: str = "datalake/bucket/clean"

        pedecerto_files: list = util.create_files_list(
            f"{raw_bucket_path}/pedecerto", "json"
        )
        pedecerto_files = [f"{raw_bucket_path}/pedecerto/" + s for s in pedecerto_files]

        hypotactic_files: list = util.create_files_list(
            f"{raw_bucket_path}/hypotactic", "json"
        )
        hypotactic_files = [
            f"{raw_bucket_path}/hypotactic/" + s for s in hypotactic_files
        ]

        all_files = pedecerto_files + hypotactic_files

        # We listed all files we think have weak meter (later or worse authors).
        # For each line, we will denote whether it is from this set of meter.
        with open('datalake/enriched/weak_poetry.txt', 'r') as file:
            weak_poetry_list = file.readlines()

        print("Building dataframe.")
        all_rows = []
        line_counter = 1  # global line number across all files
        for file_path in all_files:
            lines = util.read_json(file_path)

            for line_data in lines:
                meter = line_data["meter"]
                file_name = line_data["file_name"]
                weak_poetry = file_name in weak_poetry_list

                for entry in line_data["line"]:
                    if "-" in entry:
                        # We encode spaces a bit differently.
                        all_rows.append(
                            {
                                "meter": meter,
                                # This should be put in curated
                                "weak_poetry": weak_poetry,
                                "line_number": line_counter,
                                "syllable": "-",
                                "label": "space",
                                "word": "space",
                            }
                        )
                    else:
                        all_rows.append(
                            {
                                "meter": meter,
                                # This should be put in curated
                                "weak_poetry": weak_poetry,
                                "line_number": line_counter,
                                "syllable": entry["syllable"],
                                "label": entry["length"],
                                "word": entry["word"],
                            }
                        )

                line_counter += 1  # increment after processing each line

        # Final Polars DataFrame
        df = pl.DataFrame(all_rows)

        # Optional: inspect
        with pl.Config(tbl_rows=20):
            print(df)

        print("STATS ABOUT THIS DATAFRAME")

        df_count_grouped_by_meter = (
            df.group_by("meter")
            .agg(pl.col("line_number").n_unique().alias("unique_line_count"))
            .sort("unique_line_count", descending=True)
        )

        with pl.Config(tbl_rows=200):
            print(df_count_grouped_by_meter)

        df.write_parquet(f"{destination_path}/poetry_dataframe.parquet")
