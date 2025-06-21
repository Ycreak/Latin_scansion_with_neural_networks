To run:

```
PYTHONPATH=. python3 lsnn/clean/pedecerto/run_pedecerto.py
```

Landing zone:
Ingest all files from our sources

Raw:
Save all lines from sources in json in the following format:
{
  "lines": [
    {
      "author": "adelphoe",
      "meter": "ia6",
      "line": [
        {
          "syllable": "du",
          "word": "duos",
          "length": "short"
        },
        {
          "syllable": "os",
          "word": "duos",
          "length": "long"
        },

Enriched:
Save all lines in a clean and uniform way in Polars.

author     title    meter       line    syllable        word        length
virgil     aeneid   hexameter   1       ar              arma        long 

Curated:
Save all lines per category we want to run our tests on.


