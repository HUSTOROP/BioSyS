# Data directory

Place the HUST-SOFT CSV file and image folders in this directory. Dataset files
are intentionally excluded from Git because they are large.

The default configuration expects:

```text
data/
├── tactile_dataset_final1.csv
└── <image folders referenced by image_path in the CSV>
```

Each `image_path` should be relative to this directory. Both `/` and `\`
separators are accepted by the loader.
